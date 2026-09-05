"""Thin Travel structured-generation adapter for the stock MAGRPO trainer."""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import wandb

from comlrl.trainers.reinforce import MAGRPOTrainer
from comlrl.utils.distributed import unwrap_model

from single_turn.aggregation import ordered_owned_slots
from single_turn.formatting import build_agent_json_prefill
from single_turn.structured_generation import (
    CompleteJSONObjectCriteria,
    FixedSlotJSONLogitsProcessor,
    with_fixed_slot_json_processor,
    with_greedy_argmax_processor,
    with_json_stopping_criterion,
)


class StructuredOutputMAGRPOTrainer(MAGRPOTrainer):
    """MAGRPO trainer that stops train and eval rollouts at one JSON object."""

    def __init__(
        self,
        *args: Any,
        chat_formatted_prompts: bool = False,
        force_json_prefix: bool = True,
        constrain_json_skeleton: bool = True,
        max_value_tokens: int = 32,
        normalize_value_log_probs: bool = True,
        role_mode: str = "partitioned_roles",
        stop_after_complete_json: bool = True,
        rotate_eval_subset: bool = False,
        greedy_eval: bool = True,
        curriculum_train_dataset: Any = None,
        curriculum_short_epochs: int = 0,
        **kwargs: Any,
    ):
        self.chat_formatted_prompts = bool(chat_formatted_prompts)
        self.force_json_prefix = bool(force_json_prefix)
        self.constrain_json_skeleton = bool(constrain_json_skeleton)
        self.max_value_tokens = int(max_value_tokens)
        self.normalize_value_log_probs = bool(normalize_value_log_probs)
        self.role_mode = str(role_mode)
        self.stop_after_complete_json = bool(stop_after_complete_json)
        self.rotate_eval_subset = bool(rotate_eval_subset)
        self.greedy_eval = bool(greedy_eval)
        self.curriculum_train_dataset = curriculum_train_dataset
        self.curriculum_short_epochs = int(curriculum_short_epochs)
        if self.curriculum_short_epochs < 0:
            raise ValueError("curriculum_short_epochs must be non-negative.")
        self._travel_eval_generation = False
        self._travel_training_active = False
        self._travel_train_epoch_cursor = 0
        self._travel_curriculum_stage = 1
        self._travel_train_detail_groups: List[Dict[str, float]] = []
        self._eval_cursor = 0
        if self.max_value_tokens < 1:
            raise ValueError("max_value_tokens must be positive.")
        if self.constrain_json_skeleton and not (
            self.chat_formatted_prompts
            and self.force_json_prefix
            and self.stop_after_complete_json
        ):
            raise ValueError(
                "constrain_json_skeleton=true requires chat formatting, the JSON "
                "assistant prefix, and complete-JSON stopping."
            )
        if self.constrain_json_skeleton and self.role_mode != "partitioned_roles":
            raise ValueError(
                "constrain_json_skeleton=true currently requires "
                "role_mode=partitioned_roles."
            )
        super().__init__(*args, **kwargs)
        if self.constrain_json_skeleton and bool(
            getattr(self.args, "reference_kl_enabled", False)
        ):
            raise ValueError(
                "Travel's fixed skeleton excludes schema tokens from the policy "
                "loss, so reference_kl_enabled must remain false until the same "
                "value-token mask is applied to the reference KL."
            )
        if self.curriculum_short_epochs > int(self.args.num_train_epochs):
            raise ValueError(
                "curriculum_short_epochs cannot exceed num_train_epochs."
            )
        if self.curriculum_short_epochs and self.curriculum_train_dataset is None:
            raise ValueError(
                "curriculum_train_dataset is required when the short phase is enabled."
            )

    def train(self, **kwargs: Any) -> Any:
        """Run a Travel-only two-stage curriculum around stock MAGRPO."""

        if self._travel_training_active:
            raise RuntimeError("StructuredOutputMAGRPOTrainer.train is not reentrant.")
        self._travel_training_active = True
        self._travel_train_epoch_cursor = 0
        self._travel_curriculum_stage = 0 if self.curriculum_short_epochs else 1
        self._travel_train_detail_groups.clear()
        try:
            drain_details = getattr(
                getattr(self, "reward_func", None), "drain_details", None
            )
            if callable(drain_details):
                drain_details()
            result = super().train(**kwargs)
            expected_epochs = int(self.args.num_train_epochs)
            if self._travel_train_epoch_cursor != expected_epochs:
                raise RuntimeError(
                    "MAGRPO constructed an unexpected number of training "
                    "dataloaders; refusing to silently change the curriculum."
                )
            return result
        finally:
            self._travel_training_active = False
            self._travel_curriculum_stage = 1

    def get_train_dataloader(self) -> Any:
        """Select the short-trip dataset only for the configured first epochs."""

        if not getattr(self, "_travel_training_active", False):
            return super().get_train_dataloader()
        epoch_idx = int(self._travel_train_epoch_cursor)
        use_short = epoch_idx < int(self.curriculum_short_epochs)
        selected_dataset = (
            self.curriculum_train_dataset if use_short else self.train_dataset
        )
        original_dataset = self.train_dataset
        self._travel_curriculum_stage = 0 if use_short else 1
        try:
            self.train_dataset = selected_dataset
            dataloader = super().get_train_dataloader()
        finally:
            self.train_dataset = original_dataset
        self._travel_train_epoch_cursor += 1
        return dataloader

    def _compute_rewards(
        self, prompts: Any, completions_list: Any, batch_items: Any = None
    ) -> List[float]:
        """Capture Travel reward diagnostics without touching MAGRPO core."""

        rewards = super()._compute_rewards(
            prompts, completions_list, batch_items=batch_items
        )
        drain_details = getattr(self.reward_func, "drain_details", None)
        details = drain_details() if callable(drain_details) else []
        if not (
            getattr(self, "_travel_training_active", False)
            and not getattr(self, "_travel_eval_generation", False)
            and rewards
            and details
        ):
            return rewards

        reward_values = [float(value) for value in rewards]
        reward_mean = sum(reward_values) / len(reward_values)
        centered = [abs(value - reward_mean) for value in reward_values]
        reward_std = (
            sum((value - reward_mean) ** 2 for value in reward_values)
            / len(reward_values)
        ) ** 0.5
        record: Dict[str, float] = {
            "_step": float(self.env_step + len(reward_values)),
            "train/reward": reward_mean,
            "train/reward_group_std": reward_std,
            "train/nonzero_advantage_rate": sum(
                value > 1e-6 for value in centered
            )
            / len(centered),
            "train/meaningful_advantage_rate": sum(
                value > 1e-3 for value in centered
            )
            / len(centered),
            "train/curriculum_stage": float(
                getattr(self, "_travel_curriculum_stage", 1)
            ),
        }
        detail_keys = {
            "train/action_validity": "action_validity",
            "train/team_action_success": "ultimate/team_action_success",
            "train/required_cooperative_contribution": (
                "required_cooperative_contribution"
            ),
            "train/required_grounded_recall": "required_grounded_recall",
            "train/entity_grounding_precision": "entity_grounding_precision",
            "train/grounding_f1": "grounding_f1",
            "train/required_cost_completeness": (
                "required_cost_completeness"
            ),
            "train/reference_budget_soft": "budget_constraint_soft",
            "train/reference_budget_pass": "ultimate/reference_budget_pass",
            "train/reference_commonsense_soft": "commonsense_soft",
            "train/reference_hard_soft": "hard_constraint_soft",
            "train/required_plan_completion": (
                "ultimate/required_plan_completion"
            ),
            "train/reference_plan_success": (
                "ultimate/reference_plan_success"
            ),
            "train/plan_score": "plan_score",
            "train/strict_composite_quality": "strict_composite_quality",
            "train/protocol_progress": "protocol_progress",
            "train/recovered_semantic_balance": "recovered_semantic_balance",
            "train/recovered_plan_score": "recovered_plan_score",
            "train/recovered_composite_quality": (
                "recovered_composite_quality"
            ),
            "train/collaboration_success": "ultimate/collaboration_success",
        }
        for output_key, detail_key in detail_keys.items():
            values = [
                float(detail[detail_key])
                for detail in details
                if isinstance(detail.get(detail_key), Real)
            ]
            if values:
                record[output_key] = sum(values) / len(values)
        self._travel_train_detail_groups.append(record)
        return rewards

    def _should_log_train(self, step: int) -> bool:
        """Keep Travel's W&B history focused on fixed-anchor eval curves."""

        return False

    def _process_buffer(self, agent_idx: int, buffer: Any) -> Dict[str, Any]:
        """Attach one shared set of Travel metrics to agent 0's train log."""

        result = super()._process_buffer(agent_idx, buffer)
        if agent_idx != 0 or not result.get("log_entries"):
            return result
        pending = getattr(self, "_travel_train_detail_groups", [])
        for entry in result["log_entries"]:
            step = int(entry.get("step", 0))
            selected = [item for item in pending if item.get("_step", 0.0) <= step]
            if not selected:
                continue
            metric_keys = sorted(
                {key for item in selected for key in item if key != "_step"}
            )
            entry.setdefault("metrics", {}).update(
                {
                    key: sum(item[key] for item in selected if key in item)
                    / sum(key in item for item in selected)
                    for key in metric_keys
                }
            )
            pending = [item for item in pending if item.get("_step", 0.0) > step]
        self._travel_train_detail_groups = pending
        return result

    def _pack_completions_for_buffer(
        self, completions_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Keep Travel's value-token mask when stock MAGRPO buffers a rollout."""

        packed = super()._pack_completions_for_buffer(completions_data)
        raw_masks = completions_data.get("completion_loss_mask")
        if raw_masks:
            masks = raw_masks[0] if isinstance(raw_masks[0], list) else raw_masks
            packed["completion_loss_mask"] = [[mask.cpu() for mask in masks]]
        return packed

    def _compute_loss_with_gradients(
        self, agent: Any, completions_data: Dict[str, Any], returns: Any
    ) -> torch.Tensor:
        """Apply MAGRPO only to freely selected value tokens for this domain.

        The fixed JSON skeleton is an environment action schema, not a policy
        decision.  Excluding those forced tokens and averaging the remaining
        token log probabilities prevents long Travel JSON responses from
        scaling one policy update by hundreds of syntax tokens.
        """

        raw_masks = completions_data.get("completion_loss_mask")
        if not raw_masks:
            return super()._compute_loss_with_gradients(
                agent, completions_data, returns
            )

        agent_module = unwrap_model(agent)
        device = next(agent_module.parameters()).device
        if len(returns) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)
        effective_returns = self._apply_reference_kl_to_returns(
            returns_tensor, completions_data
        )
        advantages = self._compute_advantages(effective_returns)
        if self.args.advantage_normalization and advantages.numel() > 1:
            mean = advantages.mean()
            std = advantages.std(unbiased=False).clamp(min=1e-6)
            advantages = (advantages - mean) / std

        agent.train()
        prompt_ids = completions_data["prompt_input_ids"].to(device)[0]
        raw_completion_ids = completions_data["completion_input_ids"]
        completion_ids = (
            raw_completion_ids[0]
            if raw_completion_ids and isinstance(raw_completion_ids[0], list)
            else raw_completion_ids
        )
        masks = raw_masks[0] if isinstance(raw_masks[0], list) else raw_masks
        if len(completion_ids) != len(masks):
            raise ValueError(
                "Travel completion IDs and value-token masks must have the same "
                f"number of sequences; got {len(completion_ids)} and {len(masks)}."
            )
        if len(completion_ids) != len(advantages):
            raise ValueError(
                "Travel completions and advantages must have the same number of "
                f"sequences; got {len(completion_ids)} and {len(advantages)}."
            )

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0
        for seq_idx, (raw_tokens, raw_mask) in enumerate(
            zip(completion_ids, masks)
        ):
            completion_tokens = raw_tokens.to(device)
            loss_mask = raw_mask.to(device=device, dtype=torch.bool)
            if completion_tokens.numel() != loss_mask.numel():
                raise ValueError(
                    "Each Travel completion and value-token mask must have equal "
                    f"length; sequence {seq_idx} has {completion_tokens.numel()} "
                    f"tokens and {loss_mask.numel()} mask entries."
                )
            if completion_tokens.numel() < 1:
                continue
            input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
            attention_mask = torch.ones(len(input_ids), device=device)
            outputs = agent(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                use_cache=False,
            )
            # ``input_ids`` already omits only the final completion token, so
            # the prompt's last position through the sequence's last position
            # provides exactly one prediction for every target token.
            completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 :, :]
            if completion_logits.size(0) != completion_tokens.numel():
                raise RuntimeError(
                    "Travel policy-loss logits are not aligned with completion "
                    f"tokens: got {completion_logits.size(0)} predictions for "
                    f"{completion_tokens.numel()} targets."
                )
            selected_logits = completion_logits[loss_mask]
            selected_targets = completion_tokens[loss_mask]
            selected_count = int(selected_targets.numel())
            if selected_count < 1:
                continue
            # Match stock MAGRPO's objective: rollout temperature/top-p and the
            # JSON grammar construct actions, while replay scores chosen tokens
            # under the base LM distribution. Keeping the full-vocabulary
            # denominator also teaches the model to move probability mass away
            # from tokens the environment had to mask as invalid JSON.
            sequence_log_prob = -F.cross_entropy(
                selected_logits,
                selected_targets,
                reduction="sum",
            )
            if getattr(self, "normalize_value_log_probs", True):
                sequence_log_prob = sequence_log_prob / selected_count
            total_loss = total_loss - sequence_log_prob * advantages[seq_idx]
            num_samples += 1

        if num_samples:
            total_loss = total_loss / num_samples
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)
        return total_loss

    def _log_eval_metrics(
        self,
        all_agent_completions_turns: Any,
        all_test_cases: Any,
        all_entry_points: Any,
        all_prompts: Any,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Publish one compact, stable set of Travel eval metrics to W&B."""

        raw_metrics: Dict[str, Any] = {}
        detailed = []
        if (
            self.eval_logger is not None
            and self.eval_aggregator is not None
            and all_agent_completions_turns
            and all(agent_comps for agent_comps in all_agent_completions_turns)
        ):
            detailed = self.eval_logger(
                agent_completions_turns=all_agent_completions_turns,
                test_cases=all_test_cases,
                entry_points=all_entry_points,
                prompts=all_prompts,
            )
            anchor_size = int(getattr(self.args, "eval_num_samples", len(detailed)))
            is_full_eval = len(detailed) > anchor_size
            if is_full_eval:
                # Keep the full-pool result available to the caller, while
                # W&B receives only the same fixed anchor as earlier evals.
                full_rows = [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "_eval_sample"
                    }
                    for row in detailed
                ]
                full_aggregated = self.eval_aggregator(
                    full_rows, num_turns=self.args.num_turns
                )
                anchor_rows = detailed[:anchor_size]
                anchor_aggregated = self.eval_aggregator(
                    anchor_rows, num_turns=self.args.num_turns
                )
                raw_metrics.update(
                    {
                        f"eval_full/{key}": value
                        for key, value in full_aggregated.items()
                    }
                )
                raw_metrics.update(
                    {
                        f"eval/{key}": value
                        for key, value in anchor_aggregated.items()
                    }
                )
                anchor_reward_values = [
                    row.get("_eval_sample", {}).get("reward")
                    for row in anchor_rows
                ]
                if not all(
                    isinstance(value, Real) for value in anchor_reward_values
                ):
                    raise ValueError(
                        "Every fixed-anchor eval row must contain a numeric reward."
                    )
                anchor_rewards = [float(value) for value in anchor_reward_values]
                anchor_reward = sum(anchor_rewards) / len(anchor_rewards)
                raw_metrics["eval/turn_1/reward_mean"] = anchor_reward
            else:
                aggregated = self.eval_aggregator(
                    detailed, num_turns=self.args.num_turns
                )
                raw_metrics.update(
                    {f"eval/{key}": value for key, value in aggregated.items()}
                )
        if isinstance(extra_metrics, dict):
            if detailed and len(detailed) > int(
                getattr(self.args, "eval_num_samples", len(detailed))
            ):
                raw_metrics.update(
                    {
                        (
                            "eval_full/" + key.removeprefix("eval/")
                            if key.startswith("eval/")
                            else key
                        ): value
                        for key, value in extra_metrics.items()
                    }
                )
            else:
                raw_metrics.update(extra_metrics)

        scalar_sources = {
            "reward": "turn_1/reward_mean",
            "action_validity": "turn_1/action_validity",
            "team_action_success": "turn_1/ultimate/team_action_success",
            "required_cooperative_contribution": (
                "turn_1/required_cooperative_contribution"
            ),
            "required_grounded_recall": "turn_1/required_grounded_recall",
            "entity_grounding_precision": "turn_1/entity_grounding_precision",
            "grounding_f1": "turn_1/grounding_f1",
            "required_cost_completeness": (
                "turn_1/required_cost_completeness"
            ),
            "reference_budget_soft": "turn_1/budget_constraint_soft",
            "reference_budget_pass": (
                "turn_1/ultimate/reference_budget_pass"
            ),
            "route_scaffold_match": "turn_1/route_scaffold_match_rate",
            "reference_plan_delivery": (
                "turn_1/ultimate/reference_plan_delivery"
            ),
            "required_plan_completion": (
                "turn_1/ultimate/required_plan_completion"
            ),
            "reference_commonsense_micro": (
                "turn_1/ultimate/reference_commonsense_micro"
            ),
            "reference_hard_micro": "turn_1/ultimate/reference_hard_micro",
            "reference_plan_success": (
                "turn_1/ultimate/reference_plan_success"
            ),
            "collaboration_success": (
                "turn_1/ultimate/collaboration_success"
            ),
        }
        metrics: Dict[str, Any] = {}
        for namespace in ("eval", "eval_full"):
            for output_name, source_suffix in scalar_sources.items():
                source = f"{namespace}/{source_suffix}"
                if isinstance(raw_metrics.get(source), Real):
                    metrics[f"{namespace}/{output_name}"] = float(
                        raw_metrics[source]
                    )

        if self.wandb_initialized and wandb.run is not None:
            # One committed write avoids W&B dropping a second write at the
            # same explicit step. Full-pool results stay in the return value;
            # only the fixed-anchor scalar curves are uploaded.
            wandb_metrics = {
                key: value for key, value in metrics.items()
                if key.startswith("eval/")
            }
            if wandb_metrics:
                wandb.log(wandb_metrics, step=self.env_step, commit=True)
        return metrics

    def evaluate(self, num_eval_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a stable held-out shard while leaving stock MAGRPO untouched."""

        previous_eval_state = getattr(self, "_travel_eval_generation", False)
        self._travel_eval_generation = True
        try:
            return self._evaluate_travel_subset(num_eval_samples)
        finally:
            self._travel_eval_generation = previous_eval_state

    def _evaluate_travel_subset(
        self, num_eval_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optionally rotate the eval dataset for legacy configurations."""

        if not self.rotate_eval_subset or self.eval_dataset is None:
            return super().evaluate(num_eval_samples=num_eval_samples)
        try:
            dataset_size = len(self.eval_dataset)
        except TypeError:
            return super().evaluate(num_eval_samples=num_eval_samples)
        if dataset_size < 1:
            return super().evaluate(num_eval_samples=num_eval_samples)

        sample_count = int(
            self.args.eval_num_samples if num_eval_samples is None else num_eval_samples
        )
        if sample_count < 1:
            raise ValueError("num_eval_samples must be positive for rotating eval.")
        cursor = self._eval_cursor % dataset_size
        original_dataset = self.eval_dataset
        self.eval_dataset = [
            original_dataset[(cursor + offset) % dataset_size]
            for offset in range(dataset_size)
        ]
        try:
            metrics = super().evaluate(num_eval_samples=sample_count)
        except Exception:
            raise
        else:
            self._eval_cursor = (
                cursor + min(sample_count, dataset_size)
            ) % dataset_size
            return metrics
        finally:
            self.eval_dataset = original_dataset

    def _resolve_prompts(
        self,
        batch_items: List[Dict[str, Any]],
        *,
        agent_idx: int,
        prompts_override: Optional[List[str]],
        external_prompts: Any,
    ) -> List[str]:
        if prompts_override is not None:
            if len(prompts_override) != len(batch_items):
                raise ValueError(
                    "prompts_override must have the same length as batch_items"
                )
            return list(prompts_override)

        formatter = self.formatters[agent_idx]
        if external_prompts is None:
            return [formatter(item) for item in batch_items]

        if isinstance(external_prompts, (list, tuple)):
            if len(external_prompts) != len(batch_items):
                raise ValueError(
                    "external_prompts must have the same length as batch_items"
                )
            external_list = list(external_prompts)
        else:
            external_list = [external_prompts for _ in batch_items]
        return [
            formatter(item, external_prompts=external)
            for item, external in zip(batch_items, external_list)
        ]

    def _generate_completions(
        self,
        agent: Any,
        batch_items: List[Dict[str, Any]],
        agent_idx: int = 0,
        num_return_sequences: int = 1,
        max_new_tokens: int = 128,
        prompts_override: Optional[List[str]] = None,
        external_prompts: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompts = self._resolve_prompts(
            batch_items,
            agent_idx=agent_idx,
            prompts_override=prompts_override,
            external_prompts=external_prompts,
        )
        assistant_prefix = ""
        if self.force_json_prefix:
            if not self.chat_formatted_prompts:
                raise ValueError(
                    "travel.force_json_prefix=true requires chat-formatted prompts."
                )
            prefixes = [
                build_agent_json_prefill(agent_idx, int(item.get("days", 0)))
                for item in batch_items
            ]
            # Every generation call belongs to one agent, whose first role-owned
            # slot is invariant across Travel trip lengths. Keeping this explicit
            # prevents a future role change from silently using the wrong JSON
            # parser state for a mixed batch.
            if len(set(prefixes)) != 1:
                raise ValueError(
                    "A generation batch must use one shared Travel assistant prefill."
                )
            assistant_prefix = prefixes[0]
            # The fixed schema/prefix is context rather than a sampled policy
            # token. Reward-facing text is reconstructed after generation.
            prompts = [
                prompt + prefix for prompt, prefix in zip(prompts, prefixes)
            ]

        tokenizer = self.tokenizers[agent_idx]
        prompt_encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = int(prompt_encodings.input_ids.shape[-1])
        skeleton_processor = None
        if getattr(self, "constrain_json_skeleton", False):
            slot_orders = [
                tuple(
                    ordered_owned_slots(
                        agent_idx,
                        int(item.get("days", 0)),
                    )
                )
                for item in batch_items
            ]
            if len(set(slot_orders)) != 1:
                raise ValueError(
                    "A constrained generation batch must share one Travel slot order."
                )
            skeleton_processor = FixedSlotJSONLogitsProcessor(
                tokenizer,
                prompt_length=prompt_length,
                slots=slot_orders[0],
                max_value_tokens=int(getattr(self, "max_value_tokens", 32)),
                max_new_tokens=max_new_tokens,
            )
            kwargs["logits_processor"] = with_fixed_slot_json_processor(
                kwargs.get("logits_processor"), skeleton_processor
            )
        if (
            getattr(self, "_travel_eval_generation", False)
            and getattr(self, "greedy_eval", True)
        ):
            # Grammar masking must run before argmax so eval chooses the best
            # token that is valid in the current value/skeleton state.
            kwargs["logits_processor"] = with_greedy_argmax_processor(
                kwargs.get("logits_processor")
            )
        json_criterion = None
        if self.stop_after_complete_json:
            json_criterion = CompleteJSONObjectCriteria(
                tokenizer,
                prompt_length=prompt_length,
                initial_text=assistant_prefix,
            )
            kwargs["stopping_criteria"] = with_json_stopping_criterion(
                kwargs.get("stopping_criteria"),
                tokenizer=tokenizer,
                prompt_length=prompt_length,
                criterion=json_criterion,
            )

        # Only generation-native kwargs reach stock CoMLRL. Response cropping is
        # done locally below, so no core response-length or loss-mask hook is needed.
        result = super()._generate_completions(
            agent,
            batch_items,
            agent_idx=agent_idx,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            prompts_override=prompts,
            external_prompts=None,
            **kwargs,
        )

        prompt_ids = result["prompt_input_ids"]
        correct_prompt_width = int(prompt_ids.shape[-1])
        # Stock CoMLRL uses the first pad token as its generation boundary. Undo
        # that slice locally if a rendered Travel prompt contains the pad id.
        pad_positions = prompt_ids[0].eq(tokenizer.pad_token_id).nonzero()
        base_prompt_width = (
            int(pad_positions[0].item())
            if pad_positions.numel() > 0
            else correct_prompt_width
        )
        base_slice_offset = correct_prompt_width - base_prompt_width

        source_tokens = result["completion_input_ids"][0]
        cropped_tokens = []
        response_lens = []
        completion_texts = []
        completion_masks = []
        completed_lengths = (
            json_criterion.completed_response_lengths
            if json_criterion is not None
            else ()
        )
        for seq_idx, raw_tokens in enumerate(source_tokens):
            generated_tokens = raw_tokens[base_slice_offset:]
            resolved_length = (
                completed_lengths[seq_idx] if seq_idx < len(completed_lengths) else None
            )
            if resolved_length is None:
                resolved_length = int(generated_tokens.numel())
                pad_positions = generated_tokens.eq(tokenizer.pad_token_id).nonzero()
                if pad_positions.numel() > 0:
                    resolved_length = int(pad_positions[0].item())
            resolved_length = max(
                0, min(int(resolved_length), int(generated_tokens.numel()))
            )
            tokens = generated_tokens[:resolved_length]
            text = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if self.force_json_prefix:
                text = assistant_prefix + text
            cropped_tokens.append(tokens)
            response_lens.append(resolved_length)
            completion_texts.append(text)
            completion_masks.append(tokens.new_ones(tokens.numel()))

        if skeleton_processor is not None:
            policy_masks = skeleton_processor.finalize_loss_masks(cropped_tokens)
            result["completion_loss_mask"] = [
                [
                    tokens.new_tensor(mask, dtype=torch.bool)
                    for tokens, mask in zip(cropped_tokens, policy_masks)
                ]
            ]

        result["prompts"] = prompts
        result["completions"] = [completion_texts]
        result["completion_input_ids"] = [cropped_tokens]
        result["completion_attention_mask"] = [completion_masks]
        result["response_lens"] = response_lens
        return result
