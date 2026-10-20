"""Travel action adapters for CoMLRL's independent and shared-critic trainers."""

from collections import defaultdict

import torch

from comlrl.trainers.actor_critic import IACTrainer, MAACTrainer
from comlrl.utils.distributed import unwrap_model
from comlrl.utils.tokenizer_utils import apply_tokenizer_specials
from single_turn.aggregation import ordered_owned_slots
from single_turn.evaluation import evaluate_single_turn_response
from single_turn.formatting import build_agent_json_prefill
from single_turn.logger import _scalarize_reward_details, aggregate_single_turn_metrics
from single_turn.structured_generation import (
    CompleteJSONObjectCriteria,
    FixedSlotJSONLogitsProcessor,
    with_fixed_slot_json_processor,
    with_greedy_argmax_processor,
    with_json_stopping_criterion,
)


def travel_actor_critic_metrics(rollouts):
    groups = defaultdict(dict)
    for sample in rollouts:
        item = sample.metadata["batch_item"]
        key = (id(item), sample.metadata.get("generation_idx", 0))
        groups[key][sample.agent_idx] = sample
    records = []
    for group in groups.values():
        if set(group) != {0, 1}:
            continue
        detail = evaluate_single_turn_response(
            [group[index].completion for index in range(2)],
            batch_item=group[0].metadata["batch_item"],
        )
        records.append(
            {f"turn_1/{k}": v for k, v in _scalarize_reward_details(detail).items()}
        )
    return aggregate_single_turn_metrics(records, num_turns=1)


class TravelActorCriticMixin:
    """Keep the Travel grammar and masks around the library's AC updates."""

    def __init__(
        self,
        *args,
        max_value_tokens=32,
        normalize_value_log_probs=True,
        greedy_eval=True,
        **kwargs,
    ):
        config = kwargs.get("args")
        if (
            config is None
            or config.num_agents != 2
            or config.num_turns != 1
            or config.critic_type != "v"
        ):
            raise ValueError("Travel actor-critic requires two roles and one turn.")
        if config.reference_kl_enabled:
            raise ValueError(
                "Travel value-only policies require reference_kl_enabled=false."
            )
        if max_value_tokens < 1:
            raise ValueError("max_value_tokens must be positive.")
        self.max_value_tokens = int(max_value_tokens)
        self.normalize_value_log_probs = bool(normalize_value_log_probs)
        self.greedy_eval = bool(greedy_eval)
        self._travel_item = None
        self._travel_generated = {}
        self._travel_reward = kwargs["reward_func"]
        kwargs["reward_func"] = self._joint_rewards
        kwargs.setdefault("metrics_callback", travel_actor_critic_metrics)
        super().__init__(*args, **kwargs)
        if any(
            isinstance(module, torch.nn.Dropout) and module.p
            for actor in self.agents
            for module in actor.modules()
        ):
            raise ValueError(
                "Travel actor-critic requires actor dropout=0 (as in Qwen3)."
            )

    def _joint_rewards(self, *completions, batch_items, prompts=None):
        if len(batch_items) != 1:
            raise ValueError("Travel actor-critic collects one task at a time.")
        # The library passes one task with several aligned joint generations.
        return self._travel_reward(
            *completions,
            batch_items=list(batch_items) * len(completions[0]),
            prompts=prompts,
        )

    def _encode_prompt(self, prompt, agent_idx=None, tokenizer=None, device=None):
        tokenizer = tokenizer or self._get_tokenizer(agent_idx)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        target = device or self.device
        return {key: encoded[key].to(target) for key in ("input_ids", "attention_mask")}

    def _travel_layout(self, agent_idx, days):
        """The assistant prefill, slot order and grammar of one generation."""
        if self.is_centralized:
            # A joint actor writes its adapter's layout, as centralized MAGRPO does.
            adapter = self.centralized_collaboration.adapter
            return (
                adapter.assistant_prefill(days),
                adapter.slot_order(days),
                adapter.skeleton_processor_cls,
            )
        return (
            build_agent_json_prefill(agent_idx, days),
            ordered_owned_slots(agent_idx, days),
            FixedSlotJSONLogitsProcessor,
        )

    def _generate_travel(self, agent_model, prompt, agent_idx, num_ret):
        item = self._travel_item
        if item is None:
            raise RuntimeError("Travel generation requires a rollout item.")
        tokenizer = self._get_tokenizer(agent_idx)
        model = unwrap_model(agent_model)
        device = self.agent_devices[agent_idx]
        apply_tokenizer_specials(tokenizer, [model])
        prefix, slots, grammar_cls = self._travel_layout(agent_idx, int(item["days"]))
        encoded = self._encode_prompt(prompt + prefix, agent_idx, device=device)
        prompt_len = encoded["input_ids"].shape[1]
        backbone = getattr(model, "model", model)
        limit = getattr(backbone.config, "max_position_embeddings", None)
        if limit and prompt_len + self.args.max_new_tokens > limit:
            raise ValueError(
                "Travel prompt plus response budget exceeds the model context window."
            )
        grammar = grammar_cls(
            tokenizer,
            prompt_length=prompt_len,
            slots=slots,
            max_value_tokens=self.max_value_tokens,
            max_new_tokens=self.args.max_new_tokens,
        )
        processors = with_fixed_slot_json_processor(None, grammar)
        if getattr(self, "_in_eval", False) and self.greedy_eval:
            processors = with_greedy_argmax_processor(processors)
        criterion = CompleteJSONObjectCriteria(
            tokenizer, prompt_length=prompt_len, initial_text=prefix
        )
        generation_kwargs = dict(
            **encoded,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=True,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            use_cache=True,
            num_return_sequences=num_ret,
            num_beams=1,
            logits_processor=processors,
            stopping_criteria=with_json_stopping_criterion(
                None, tokenizer=tokenizer, prompt_length=prompt_len, criterion=criterion
            ),
        )
        if self.args.top_k is not None:
            generation_kwargs["top_k"] = self.args.top_k
        training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                generated = model.generate(**generation_kwargs)
        finally:
            model.train(training)
        sequences = generated.clone()
        lengths = []
        tokens = []
        for index, sequence in enumerate(sequences):
            response = sequence[prompt_len:]
            completed = criterion.completed_response_lengths
            length = completed[index] if index < len(completed) else None
            if length is None:
                padding = response.eq(tokenizer.pad_token_id).nonzero()
                length = int(padding[0].item()) if padding.numel() else response.numel()
            lengths.append(int(length))
            tokens.append(response[:length])
        masks = [
            torch.tensor(mask, dtype=torch.bool)
            for mask in grammar.finalize_loss_masks(tokens)
        ]
        attention = torch.ones_like(sequences)
        for index, length in enumerate(lengths):
            attention[index, prompt_len + length :] = 0
        result = dict(
            prompt=prompt + prefix,
            prompt_len=prompt_len,
            sequences=sequences,
            attention_mask=attention,
            response_lens=lengths,
            reference_kls=[],
            completions=[
                prefix
                + tokenizer.decode(
                    token, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for token in tokens
            ],
            completion_loss_masks=masks,
        )
        self._travel_generated[agent_idx] = result
        return result

    def _collect_rollouts(self, item):
        self._travel_item = item
        self._travel_generated = {}
        try:
            rollouts = super()._collect_rollouts(item)
            for sample in rollouts:
                data = self._travel_generated[sample.agent_idx]
                index = sample.metadata["generation_idx"]
                mask = data["completion_loss_masks"][index]
                sample.completion = data["completions"][index]
                sample.prompt = data["prompt"]
                sample.metadata["completion_loss_mask"] = mask
                device = self.agent_devices[sample.agent_idx]
                with torch.no_grad():
                    logprob, _ = self._policy_eval(
                        self.agents[sample.agent_idx],
                        sample.full_input_ids.to(device)[None],
                        sample.attention_mask.to(device)[None],
                        sample.prompt_len,
                        sample.response_len,
                        output_values=False,
                        completion_loss_mask=mask,
                    )
                sample.old_logprob = logprob.squeeze(0).detach().cpu()
            return rollouts
        finally:
            self._travel_item = None
            self._travel_generated.clear()
            drain = getattr(self._travel_reward, "drain_details", None)
            if callable(drain):
                drain()

    def _compute_sequence_stats(
        self, sequences, logits, prompt_len, response_len, completion_loss_mask=None
    ):
        result = super()._compute_sequence_stats(
            sequences,
            logits,
            prompt_len,
            response_len,
            completion_loss_mask=completion_loss_mask,
        )
        if completion_loss_mask is not None and self.normalize_value_log_probs:
            result = result / completion_loss_mask.sum().to(result.device).clamp(min=1)
        return result

    def evaluate(self, num_eval_samples=None):
        if self.eval_dataset is None:
            return {}
        count = (
            self.args.eval_num_samples
            if num_eval_samples is None
            else int(num_eval_samples)
        )
        rollouts = []
        previous = getattr(self, "_in_eval", False)
        self._in_eval = True
        try:
            with torch.no_grad():
                for index, item in enumerate(self.eval_dataset):
                    if index >= count:
                        break
                    rollouts.extend(self._collect_rollouts(item))
        finally:
            self._in_eval = previous

        def summarize(samples, namespace):
            aliases = {
                "budget_constraint_soft": "reference_budget_soft",
                "route_scaffold_match_rate": "route_scaffold_match",
            }
            metrics = {}
            # A joint actor's samples are split into the two roles first.
            role_metrics = self._role_metrics_callback(travel_actor_critic_metrics)
            for key, value in role_metrics(samples).items():
                key = key.removeprefix("turn_1/").removeprefix("ultimate/")
                metrics[f"{namespace}/{aliases.get(key, key)}"] = value
            rewards = [
                float(sample.reward.item())
                for sample in samples
                if sample.agent_idx == 0
            ]
            if rewards:
                metrics[f"{namespace}/reward"] = sum(rewards) / len(rewards)
            return metrics

        anchor = rollouts[: self.num_agents * self.args.eval_num_samples]
        metrics = summarize(anchor, "eval")
        self._log_metrics(metrics)
        if count > self.args.eval_num_samples:
            metrics = {**metrics, **summarize(rollouts, "eval_full")}
        return metrics


class TravelIACTrainer(TravelActorCriticMixin, IACTrainer):
    def _generate_rollout(self, agent_model, prompt, agent_idx, num_ret):
        data = self._generate_travel(agent_model, prompt, agent_idx, num_ret)
        critic = (
            self.critics[agent_idx] if self.args.use_separate_critic else agent_model
        )
        device = self._critic_device(agent_idx)
        with torch.no_grad():
            data["values"] = self._value_for_critic_type(
                critic,
                data["sequences"].to(device),
                data["attention_mask"].to(device),
                data["prompt_len"],
                data["response_lens"],
            )
            data["logprobs"] = [
                self._policy_eval(
                    agent_model,
                    sequence[None],
                    attention[None],
                    data["prompt_len"],
                    length,
                    output_values=False,
                )[0].squeeze(0)
                for sequence, attention, length in zip(
                    data["sequences"], data["attention_mask"], data["response_lens"]
                )
            ]
        return data


class TravelMAACTrainer(TravelActorCriticMixin, MAACTrainer):
    def _generate(self, agent_model, prompt, agent_idx):
        count = 1 if getattr(self, "_in_eval", False) else self.args.num_generations
        return self._generate_travel(agent_model, prompt, agent_idx, count)
