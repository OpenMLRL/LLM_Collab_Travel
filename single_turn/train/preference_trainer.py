"""Travel adapters for CoMLRL's MAPL preference algorithms.

The preference objective and replay selection come from CoMLRL. This adapter
preserves the generated Travel action tokens/masks and schedules model memory.
"""

from __future__ import annotations

import gc
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoModel, AutoTokenizer

from comlrl.trainers.pbrl import (
    MADPOTrainer,
    MADPOIterTrainer,
    MARLHFIterTrainer,
    MARLHFTrainer,
)
from comlrl.trainers.pbrl.madpo import AgentPreferenceTensors, PreferencePair
from comlrl.utils.checkpoint import register_checkpoint_record
from comlrl.utils.distributed import unwrap_model
from comlrl.utils.tokenizer_utils import ensure_pad_token
from single_turn.train.structured_trainer import StructuredOutputMAGRPOTrainer
from single_turn.formatting import build_agent_json_prefill
from single_turn.centralized_comparator import project_role_tokens
from single_turn.structured_generation import wrap_formatter_with_chat_template


@register_checkpoint_record
@dataclass
class TravelPreferenceTensors(AgentPreferenceTensors):
    winner_loss_mask: torch.Tensor
    loser_loss_mask: torch.Tensor


def _move_state(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_state(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state(item, device) for item in value)
    return value


class TravelPreferenceMixin(StructuredOutputMAGRPOTrainer):
    """Share Travel generation, compact eval/iteration logs, and GPU residency."""

    def __init__(
        self,
        *args,
        preference_generation_batch_size: int = 20,
        offload_inactive_actors: bool = True,
        **kwargs,
    ):
        self.preference_generation_batch_size = int(preference_generation_batch_size)
        if self.preference_generation_batch_size < 1:
            raise ValueError("preference_generation_batch_size must be positive.")
        self.offload_inactive_actors = bool(offload_inactive_actors)
        self._preference_trajectories = None
        self.preference_joint_candidates = 0
        self.preference_pairs_generated = 0
        super().__init__(*args, **kwargs)
        if self.args.parallel_training != "none":
            raise ValueError("Travel MAPL currently requires parallel_training=none.")
        if self.curriculum_short_epochs:
            raise ValueError(
                "Travel MAPL uses the full split; curriculum_short_epochs must be 0."
            )
        for actor in self.agents:
            if any(
                isinstance(module, nn.Dropout) and module.p
                for module in actor.modules()
            ):
                raise ValueError(
                    "Travel's memory-bounded DPO requires dropout=0 (as in Qwen3)."
                )

    def train(self, **kwargs):
        # Preference trainers own their refresh/optimization loops. The
        # MAGRPO-only curriculum wrapper must not count preference dataloaders
        # as online RL epochs.
        if int(self.args.eval_interval) > 0:
            self.evaluate(num_eval_samples=int(self.args.eval_num_samples))
        result = self._train_preference_algorithm(**kwargs)
        # The counter covers this process only; checkpoint progress covers the
        # whole run, including segments before a resume. MADPO keeps its pairs,
        # the iterative trainers count them, and MARLHF records online-RL
        # progress only after training its reward model.
        progress = getattr(self, "_checkpoint_progress", None) or {}
        trained = bool(
            progress.get("preference_pairs")
            or progress.get("total_pairs")
            or "rl" in progress
        )
        if self.preference_pairs_generated == 0 and not trained:
            raise RuntimeError(
                "No non-tied Travel preferences were generated; no training occurred. Increase preference_num_candidates or inspect the candidate rewards."
            )
        return result

    def evaluate(self, num_eval_samples=None):
        # The offline preference/RM phases do not change actors. CoMLRL also
        # evaluates at the start of online training, so reuse that identical
        # baseline instead of emitting a second W&B write at env_step=0.
        count = (
            self.args.eval_num_samples if num_eval_samples is None else num_eval_samples
        )
        key = (int(self.env_step), int(count))
        if getattr(self, "_travel_last_eval_key", None) == key:
            return dict(self._travel_last_eval_metrics)
        metrics = super().evaluate(num_eval_samples=num_eval_samples)
        self._travel_last_eval_key = key
        self._travel_last_eval_metrics = dict(metrics)
        return metrics

    def _generate_completions(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        **kwargs,
    ):
        if len(batch_items) != 1:
            raise ValueError("Travel MAPL generation requires one prompt at a time.")
        result = None
        for start in range(
            0, num_return_sequences, self.preference_generation_batch_size
        ):
            count = min(
                self.preference_generation_batch_size, num_return_sequences - start
            )
            chunk = super()._generate_completions(
                agent,
                batch_items,
                agent_idx=agent_idx,
                num_return_sequences=count,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            if result is None:
                result = chunk
            else:
                for key in (
                    "completions",
                    "completion_input_ids",
                    "completion_attention_mask",
                    "completion_loss_mask",
                ):
                    result[key][0].extend(chunk[key][0])
                result["response_lens"].extend(chunk["response_lens"])
                result["reference_kls"].extend(chunk.get("reference_kls", []))
        if result is None:
            raise ValueError("num_return_sequences must be positive.")
        if self._preference_trajectories is not None:
            prompt = result["prompts"][0]
            for text, tokens, mask in zip(
                result["completions"][0],
                result["completion_input_ids"][0],
                result["completion_loss_mask"][0],
            ):
                # Generation includes the assistant prefill in the prompt but
                # reconstructs it in reward-facing text. Never tokenize that
                # full text again as a continuation (it duplicates the prefix).
                self._preference_trajectories[(agent_idx, prompt, text)] = (
                    result["prompt_input_ids"][0].detach().cpu().clone(),
                    tokens.detach().cpu().clone(),
                    mask.detach().cpu().clone(),
                )
            if agent_idx == 0:
                self.preference_joint_candidates += num_return_sequences
        return result

    def _preference_tensors_from_text(
        self, agent_idx, prompt, winner_completion, loser_completion
    ):
        if self._preference_trajectories is None:
            raise ValueError(
                "Travel preferences require recorded generation tokens and masks."
            )
        winner = self._preference_trajectories[(agent_idx, prompt, winner_completion)]
        loser = self._preference_trajectories[(agent_idx, prompt, loser_completion)]
        if not torch.equal(winner[0], loser[0]):
            raise ValueError(
                "Travel preference candidates must use the same role prompt."
            )
        return TravelPreferenceTensors(
            winner[0], winner[1], loser[1], winner[2], loser[2]
        )

    def _generate_preference_pairs_for_item(self, batch_item, **kwargs):
        self._preference_trajectories = {}
        try:
            pairs = super()._generate_preference_pairs_for_item(batch_item, **kwargs)
            for pair in pairs:
                pair.agent_tensors = [
                    self._preference_tensors_from_text(
                        idx,
                        pair.prompts[idx],
                        pair.winner_completions[idx],
                        pair.loser_completions[idx],
                    )
                    for idx in range(self.num_agents)
                ]
            self.preference_pairs_generated += len(pairs)
            return pairs
        finally:
            self._preference_trajectories = None
            drain = getattr(self.reward_func, "drain_details", None)
            if callable(drain):
                drain()

    def _validate_model_comparator_vocabulary(self):
        # Structured Travel actions and replay masks use the actors' token IDs.
        # Reject an incompatible external vocabulary before loading its weights.
        source = self.args.comparator_model_name
        verified = getattr(self, "_travel_verified_comparator_tokenizers", set())
        if source not in verified:
            comparator_tokenizer = AutoTokenizer.from_pretrained(source)
            vocabulary = comparator_tokenizer.get_vocab()
            if any(
                tokenizer.get_vocab() != vocabulary for tokenizer in self.tokenizers
            ):
                raise ValueError(
                    "Travel comparator and actors must have identical token-to-ID vocabularies "
                    "to preserve constrained actions and preference token masks."
                )
            self._travel_verified_comparator_tokenizers = verified | {source}

    def _load_comparator_agents(self):
        self._validate_model_comparator_vocabulary()
        source = self.args.comparator_model_name
        devices = self._resolve_comparator_devices()
        if len(set(devices)) == 1:
            # With parallel_training=none the two role calls are sequential.
            # A frozen external checkpoint has identical weights for both roles;
            # sharing them saves VRAM without sharing prompts or sampled actions.
            frozen = self._load_single_frozen_policy_agent(source, 0)
            return [frozen] * self.num_agents
        return super()._load_comparator_agents()

    def _get_centralized_comparator_agent(self):
        if self.args.comparator_policy == "model":
            self._validate_model_comparator_vocabulary()
        return super()._get_centralized_comparator_agent()

    def _build_centralized_comparator_prompt(self, batch_item):
        adapter = self.centralized_comparator_adapter
        raw = adapter.build_prompt(batch_item, [])
        formatter = wrap_formatter_with_chat_template(
            lambda item: raw, self.tokenizers[0], system_prompt=adapter.system_prompt
        )
        return formatter(batch_item)

    def _generate_centralized_comparator_outputs_for_item(
        self, batch_item, *, iteration_idx, num_candidates, **kwargs
    ):
        if self.args.comparator_policy != "history":
            return super()._generate_centralized_comparator_outputs_for_item(
                batch_item,
                iteration_idx=iteration_idx,
                num_candidates=num_candidates,
                **kwargs,
            )
        checkpoint = self._history_policy_checkpoint_path(iteration_idx)
        key = (int(iteration_idx), checkpoint)
        if getattr(self, "_travel_central_history_key", None) != key:
            self._clear_travel_central_history()
            self._travel_central_history_agent = (
                self._load_single_policy_checkpoint_agent(checkpoint, 0)
            )
            self._travel_central_history_key = key
        return self._generate_centralized_policy_output_for_agent(
            self._travel_central_history_agent,
            batch_item,
            prompt=self._build_centralized_comparator_prompt(batch_item),
            num_candidates=num_candidates,
            **kwargs,
        )

    def _clear_travel_central_history(self):
        agent = getattr(self, "_travel_central_history_agent", None)
        self._travel_central_history_agent = None
        self._travel_central_history_key = None
        if agent is not None:
            self._clear_transient_comparator_agents([agent])

    def _generate_centralized_policy_output_for_agent(
        self, policy_agent, batch_item, *, prompt, num_candidates, **kwargs
    ):
        if self._centralized_comparator_agent_index() != 0:
            raise ValueError("Travel centralized comparator uses policy agent index 0.")
        # The adapter fixes the joint answer's prefill, slot order and grammar.
        adapter = self.centralized_comparator_adapter
        days = batch_item["days"]
        prefill = adapter.assistant_prefill(days)
        # The joint teacher gets the role calls' aggregate budget unless the
        # config sets its own.
        budget = getattr(
            self.args, "comparator_centralized_max_new_tokens", None
        ) or self.num_agents * int(self.args.max_new_tokens)
        input_length = len(
            self.tokenizers[0].encode(prompt + prefill, add_special_tokens=False)
        )
        model_limit = getattr(
            unwrap_model(policy_agent).config, "max_position_embeddings", None
        )
        if model_limit and input_length + budget > int(model_limit):
            raise ValueError(
                "Centralized Travel prompt plus output budget exceeds the model context window; refusing truncation."
            )
        all_texts = []
        # Park unused actors/optimizer states while the single teacher generates.
        unused = [
            i
            for i, actor in enumerate(self.agents)
            if unwrap_model(actor) is not unwrap_model(policy_agent)
        ]
        with self._offload_actors(unused), self._comparator_rng(policy_agent):
            for start in range(
                0, num_candidates, self.preference_generation_batch_size
            ):
                count = min(
                    self.preference_generation_batch_size, num_candidates - start
                )
                result = StructuredOutputMAGRPOTrainer._generate_completions(
                    self,
                    policy_agent,
                    [batch_item],
                    agent_idx=0,
                    num_return_sequences=count,
                    max_new_tokens=budget,
                    prompts_override=[prompt],
                    slot_order_override=adapter.slot_order(days),
                    assistant_prefill_override=prefill,
                    skeleton_processor_cls=adapter.skeleton_processor_cls,
                    **kwargs,
                )
                all_texts.extend(result["completions"][0])
        self.preference_joint_candidates += len(all_texts)
        outputs = [
            {
                "prompts": [
                    self.formatters[i](batch_item)
                    + build_agent_json_prefill(i, batch_item["days"])
                ],
                "batch_items": [batch_item],
                "completions": [[]],
            }
            for i in range(2)
        ]
        for joint_text in all_texts:
            split = adapter.parse_completion(joint_text, batch_item, self.num_agents)
            for i, role_text in enumerate(split):
                text, tokens, mask = project_role_tokens(
                    self.tokenizers[i], role_text, i, batch_item["days"]
                )
                role_prompt = outputs[i]["prompts"][0]
                outputs[i]["completions"][0].append(text)
                if self._preference_trajectories is not None:
                    encoded = self.tokenizers[i](
                        [role_prompt],
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                    # Identical target text keeps its original, exact sampled trajectory.
                    self._preference_trajectories.setdefault(
                        (i, role_prompt, text),
                        (encoded.input_ids[0].detach().cpu().clone(), tokens, mask),
                    )
        return outputs

    def _generate_history_policy_outputs_for_item(
        self,
        batch_item,
        *,
        iteration_idx,
        num_candidates,
        **kwargs,
    ):
        checkpoint = self._history_policy_checkpoint_path(iteration_idx)
        key = (int(iteration_idx), checkpoint)
        if getattr(self, "_travel_history_comparator_key", None) != key:
            self._clear_travel_history_comparator()
            self._travel_history_comparator_agents = (
                self._load_policy_checkpoint_agents(checkpoint)
            )
            self._travel_history_comparator_key = key
        # Load once per collection phase, not twice per question. Both distinct
        # role checkpoints and CoMLRL's history-k resolution are preserved.
        return self._generate_policy_outputs_for_item(
            self._travel_history_comparator_agents,
            batch_item,
            num_candidates=num_candidates,
            use_comparator_rng=True,
            **kwargs,
        )

    def _clear_travel_history_comparator(self):
        agents = getattr(self, "_travel_history_comparator_agents", None)
        self._travel_history_comparator_agents = None
        self._travel_history_comparator_key = None
        if agents is not None:
            self._clear_transient_comparator_agents(agents)

    def _clear_iteration_current_copy_comparator(self):
        # Both iterative trainers call this in their collection-phase finally
        # block, including on errors, before fitting the RM or updating actors.
        try:
            super()._clear_iteration_current_copy_comparator()
        finally:
            self._clear_travel_history_comparator()
            self._clear_travel_central_history()

    @staticmethod
    def _preference_pair_to_record(pair):
        record = MADPOIterTrainer._preference_pair_to_record(pair)
        record["travel_tensor_version"] = 1
        record["travel_agent_tensors"] = [
            {
                key: getattr(tensors, key).tolist()
                for key in TravelPreferenceTensors.__dataclass_fields__
            }
            for tensors in pair.agent_tensors
        ]
        return record

    def _preference_pair_from_record(self, record):
        if record.get("travel_tensor_version") != 1:
            raise ValueError(
                "Replay is missing Travel token masks; use a fresh preference_replay_dir."
            )
        rows = record["travel_agent_tensors"]
        if len(rows) != self.num_agents:
            raise ValueError("Replay must contain one token/mask record per agent.")
        tensors = [
            TravelPreferenceTensors(
                **{
                    key: torch.tensor(
                        row[key],
                        dtype=torch.bool if key.endswith("loss_mask") else torch.long,
                    )
                    for key in TravelPreferenceTensors.__dataclass_fields__
                }
            )
            for row in rows
        ]
        return PreferencePair(
            prompts=list(record["prompts"]),
            winner_completions=list(record["winner_completions"]),
            loser_completions=list(record["loser_completions"]),
            agent_tensors=tensors,
            winner_reward=float(record["winner_reward"]),
            loser_reward=float(record["loser_reward"]),
            candidate_reward_mean=float(record["candidate_reward_mean"]),
            raw_rewards=record.get("raw_rewards"),
            target_raw_reward=record.get("target_raw_reward"),
            comparator_raw_reward=record.get("comparator_raw_reward"),
        )

    def _sequence_log_prob(
        self, agent_idx, prompt_input_ids, completion_ids, loss_mask=None
    ):
        if loss_mask is None or completion_ids.numel() != loss_mask.numel():
            raise ValueError(
                "Travel DPO requires one value-token mask entry per completion token."
            )
        agent = unwrap_model(self.agents[agent_idx])
        device = next(agent.parameters()).device
        prompt = prompt_input_ids.to(device)
        completion = completion_ids.to(device)
        mask = loss_mask.to(device=device, dtype=torch.bool)
        if not completion.numel() or not mask.any():
            return torch.zeros((), device=device, requires_grad=True)
        agent.train()
        inputs = torch.cat((prompt, completion[:-1])).unsqueeze(0)
        outputs = agent(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False
        )
        logits = outputs.logits[0, prompt.numel() - 1 :]
        if logits.shape[0] != completion.numel():
            raise ValueError(
                "Travel preference logits are not aligned with all target tokens."
            )
        logprob = -F.cross_entropy(logits[mask], completion[mask], reduction="sum")
        if self.normalize_value_log_probs:
            logprob = logprob / mask.sum()
        return logprob

    def _agent_logprob_delta(self, agent_idx, pair):
        tensors = pair.agent_tensors[agent_idx]
        return self._sequence_log_prob(
            agent_idx,
            tensors.prompt_input_ids,
            tensors.winner_completion_ids,
            tensors.winner_loss_mask,
        ) - self._sequence_log_prob(
            agent_idx,
            tensors.prompt_input_ids,
            tensors.loser_completion_ids,
            tensors.loser_loss_mask,
        )

    @contextmanager
    def _offload_actors(self, indices):
        parked = []
        try:
            if self.offload_inactive_actors:
                for idx in indices:
                    actor = unwrap_model(self.agents[idx])
                    device = next(actor.parameters()).device
                    if device.type != "cuda":
                        continue
                    optimizer = self.optimizers[idx]
                    optimizer.zero_grad(set_to_none=True)
                    actor.to("cpu")
                    for param, state in list(optimizer.state.items()):
                        optimizer.state[param] = _move_state(state, torch.device("cpu"))
                    parked.append((idx, device))
                if parked:
                    torch.cuda.empty_cache()
            yield
        finally:
            for idx, device in parked:
                self.agents[idx].to(device)
                optimizer = self.optimizers[idx]
                for param, state in list(optimizer.state.items()):
                    # Adam step counters normally remain on the CPU.
                    optimizer.state[param] = {
                        key: value if key == "step" else _move_state(value, device)
                        for key, value in state.items()
                    }

    def _process_buffer(self, agent_idx, buffer):
        with self._offload_actors(
            idx for idx in range(self.num_agents) if idx != agent_idx
        ):
            result = super()._process_buffer(agent_idx, buffer)
        self._travel_last_eval_key = None
        return result

    def _update_from_preference_batch(self, batch):
        if not batch:
            return {}
        self._travel_last_eval_key = None
        # Exact gradient of CoMLRL's joint -logsigmoid(beta * sum(delta)).
        # Replay winner and loser separately so only one forward graph lives
        # on the GPU, even when the optimizer batch contains several pairs.
        deltas = self._detached_agent_deltas(batch)
        beta = float(self.args.dpo_beta)
        coefficients = [
            -beta * torch.sigmoid(torch.tensor(-beta * sum(row))).item() / len(batch)
            for row in deltas
        ]
        for idx in range(self.num_agents):
            with self._offload_actors(
                other for other in range(self.num_agents) if other != idx
            ):
                optimizer = self.optimizers[idx]
                optimizer.zero_grad(set_to_none=True)
                for pair, coefficient in zip(batch, coefficients):
                    tensors = pair.agent_tensors[idx]
                    for sign, tokens, mask in (
                        (1.0, tensors.winner_completion_ids, tensors.winner_loss_mask),
                        (-1.0, tensors.loser_completion_ids, tensors.loser_loss_mask),
                    ):
                        logprob = self._sequence_log_prob(
                            idx, tensors.prompt_input_ids, tokens, mask
                        )
                        (coefficient * sign * logprob).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        mean = sum(pair.candidate_reward_mean for pair in batch) / len(batch)
        return {"turn_1/reward_mean": mean, "turn_1/expected_return": mean}

    def _log_iteration_replay(
        self, iteration_idx, *, train_pairs, current_pair_count, train_pair_count
    ):
        # Reuse CoMLRL's raw-reward bins/plots and selected-pair diagnostics,
        # but do not commit with step=env_step: refresh and eval can share it.
        distributions = self._log_reward_distribution_enabled()
        if distributions:
            self._write_iteration_reward_distribution(iteration_idx, train_pairs)
        metrics = {
            "iter/current_iteration": int(iteration_idx + 1),
            "iter/current_preference_pairs": int(current_pair_count),
            "iter/total_preference_pairs": sum(
                shard.num_pairs
                for shard in getattr(self, "_preference_replay_shards_state", [])
            ),
            "iter/train_preference_pairs": int(train_pair_count),
        }
        if self.wandb_initialized and wandb.run is not None:
            if distributions:
                metrics.update(
                    self._iteration_reward_distribution_metrics(iteration_idx)
                )
                metrics.update(
                    self._selected_reward_distribution_metrics(
                        iteration_idx, train_pairs
                    )
                )
            self._log_mapl_metrics(metrics)
        if self.verbose:
            print(
                f"Travel iteration {iteration_idx + 1}: new_pairs={current_pair_count}, replay_pairs={train_pair_count}, preference_joint_candidates={self.preference_joint_candidates}"
            )

    def _log_wandb_eval_metrics(self, metrics):
        self._log_mapl_metrics(metrics)

    def _reward_distribution_dir(self):
        # CoMLRL otherwise falls back to the source checkout when W&B is off.
        # Keep distribution artifacts with this run's isolated replay instead.
        if self.wandb_config is not None:
            return super()._reward_distribution_dir()
        path = Path(self._preference_replay_dir()).parent / "reward_distributions"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _log_mapl_metrics(self, metrics):
        if not self.wandb_initialized or wandb.run is None:
            return
        allowed = {
            key: value
            for key, value in metrics.items()
            if (key.startswith("eval/") or key.startswith("iter/"))
            and key != "eval/samples"
        }
        if not allowed:
            return
        if not getattr(self, "_travel_mapl_axes_defined", False):
            # Keep the x-axis in history, without its own plot or summary card.
            wandb.define_metric("env_step", hidden=True, summary="none")
            wandb.define_metric("eval/*", step_metric="env_step")
            if hasattr(self.args, "num_iterations"):
                wandb.define_metric("iter/current_iteration", hidden=True)
                wandb.define_metric("iter/*", step_metric="iter/current_iteration")
            self._travel_mapl_axes_defined = True
        # W&B chooses a monotonically increasing internal history step. Explicit
        # axes keep eval on env steps and iteration diagnostics on iterations,
        # including a baseline and refresh that both occur at env_step=0.
        wandb.log({"env_step": int(self.env_step), **allowed}, commit=True)


class TravelJointRewardModel(nn.Module):
    """Decoder plus scalar head, without vocabulary logits/all-layer states."""

    def __init__(self, backbone, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        hidden_size = (
            getattr(backbone.config, "hidden_size", None) or backbone.config.n_embd
        )
        reference = next(backbone.parameters())
        self.reward_head = nn.Linear(hidden_size, 1).to(
            device=reference.device, dtype=reference.dtype
        )
        if freeze_backbone:
            backbone.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        positions = attention_mask.long().sum(-1).clamp(min=1) - 1
        return self.reward_head(
            hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        ).squeeze(-1)


class TravelRewardModelMixin:
    def _init_reward_model(self):
        source = self.args.reward_model_name or self.model_name
        self.reward_tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(source))
        self.reward_tokenizer.padding_side = "right"
        kwargs = self._model_kwargs_from_config()
        if isinstance(kwargs.get("torch_dtype"), str):
            kwargs["torch_dtype"] = getattr(torch, kwargs["torch_dtype"])
        # Load on CPU. Actors are parked before this model moves to the GPU.
        backbone = AutoModel.from_pretrained(source, **kwargs)
        if not self.args.reward_freeze_backbone:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.reward_model = TravelJointRewardModel(
            backbone, self.args.reward_freeze_backbone
        )
        self.reward_optimizer = torch.optim.AdamW(
            (param for param in self.reward_model.parameters() if param.requires_grad),
            lr=float(self.args.reward_learning_rate),
        )

    def _train_reward_model(self, preference_pairs):
        with self._offload_actors(range(self.num_agents)):
            self.reward_model.to(self.reward_device)
            super()._train_reward_model(preference_pairs)
            # Online RL only evaluates the fitted RM; its Adam state and
            # gradients must not stay resident alongside both actor optimizers.
            self.reward_optimizer.zero_grad(set_to_none=True)
            self.reward_optimizer = None
            self.reward_model.requires_grad_(False)
            self.reward_model.eval()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _format_joint_text(self, prompts, completions):
        # Use exactly the same representation for preference fitting and online
        # scoring. Both complete agent actions precede the (long) role context.
        # This scorer never reads annotated_plan/gold_plan or a target reward.
        contexts = []
        for idx, prompt in enumerate(prompts):
            # The prefill is independent of trip length. Remove only a trailing
            # prefill, never JSON examples embedded in the role instructions.
            prefix = self._travel_assistant_prefill(idx, {"days": 3})
            contexts.append(
                prompt[: -len(prefix)] if prompt.endswith(prefix) else prompt
            )
        return json.dumps(
            {
                "joint_actions": list(completions),
                "role_contexts": contexts,
            },
            ensure_ascii=False,
        )

    def _score_reward_texts(self, texts):
        if self.reward_model is None or self.reward_tokenizer is None:
            raise RuntimeError("Reward model has not been initialized.")
        scores = []
        for text in texts:
            encoded = self.reward_tokenizer(text, truncation=False, return_tensors="pt")
            length = int(encoded["input_ids"].shape[-1])
            if self.args.reward_max_length and length > self.args.reward_max_length:
                raise ValueError(
                    f"Travel reward input has {length} tokens, exceeding reward_max_length={self.args.reward_max_length}; increase the limit instead of truncating an agent's action/context."
                )
            scores.append(
                self.reward_model(
                    input_ids=encoded["input_ids"].to(self.reward_device),
                    attention_mask=encoded["attention_mask"].to(self.reward_device),
                )
            )
        return torch.cat(scores)


class TravelMADPOTrainer(TravelPreferenceMixin, MADPOTrainer):
    _train_preference_algorithm = MADPOTrainer.train


class TravelMADPOIterTrainer(TravelPreferenceMixin, MADPOIterTrainer):
    _train_preference_algorithm = MADPOIterTrainer.train


class TravelMARLHFTrainer(TravelRewardModelMixin, TravelPreferenceMixin, MARLHFTrainer):
    _train_preference_algorithm = MARLHFTrainer.train


class TravelMARLHFIterTrainer(
    TravelRewardModelMixin, TravelPreferenceMixin, MARLHFIterTrainer
):
    _train_preference_algorithm = MARLHFIterTrainer.train
