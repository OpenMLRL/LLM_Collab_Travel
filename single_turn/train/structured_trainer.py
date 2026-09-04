"""Thin Travel structured-generation adapter for the stock MAGRPO trainer."""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List, Optional

import wandb

from comlrl.trainers.reinforce import MAGRPOTrainer

from single_turn.formatting import build_agent_json_prefill
from single_turn.structured_generation import (
    CompleteJSONObjectCriteria,
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
        stop_after_complete_json: bool = True,
        rotate_eval_subset: bool = False,
        greedy_eval: bool = True,
        curriculum_train_dataset: Any = None,
        curriculum_short_epochs: int = 0,
        **kwargs: Any,
    ):
        self.chat_formatted_prompts = bool(chat_formatted_prompts)
        self.force_json_prefix = bool(force_json_prefix)
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
        self._travel_eval_baseline: Optional[Dict[str, float]] = None
        self._eval_cursor = 0
        super().__init__(*args, **kwargs)
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
            "train/reference_commonsense_soft": "commonsense_soft",
            "train/reference_hard_soft": "hard_constraint_soft",
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

    def _log_eval_metrics(
        self,
        all_agent_completions_turns: Any,
        all_test_cases: Any,
        all_entry_points: Any,
        all_prompts: Any,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log Travel-friendly aliases without changing stock MAGRPO.

        Existing W&B workspaces often point at ``eval/reward`` while modern
        MAGRPO emits ``eval/turn_1/reward_mean``.  Keep both names, and expose a
        compact set of headline aliases next to the detailed metric tree.
        """

        metrics: Dict[str, Any] = {}
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
                full_aggregated = self.eval_aggregator(
                    detailed, num_turns=self.args.num_turns
                )
                anchor_rows = detailed[:anchor_size]
                anchor_aggregated = self.eval_aggregator(
                    anchor_rows, num_turns=self.args.num_turns
                )
                metrics.update(
                    {
                        f"eval_full/{key}": value
                        for key, value in full_aggregated.items()
                    }
                )
                metrics.update(
                    {
                        f"eval/{key}": value
                        for key, value in anchor_aggregated.items()
                    }
                )
                anchor_rewards = [
                    row.get("_eval_sample", {}).get("reward")
                    for row in anchor_rows
                ]
                anchor_rewards = [
                    float(value)
                    for value in anchor_rewards
                    if isinstance(value, Real)
                ]
                if anchor_rewards:
                    anchor_reward = sum(anchor_rewards) / len(anchor_rewards)
                    metrics["eval/turn_1/reward_mean"] = anchor_reward
                    metrics["eval/turn_1/expected_return"] = anchor_reward
            else:
                aggregated = self.eval_aggregator(
                    detailed, num_turns=self.args.num_turns
                )
                metrics.update(
                    {f"eval/{key}": value for key, value in aggregated.items()}
                )
        if isinstance(extra_metrics, dict):
            if detailed and len(detailed) > int(
                getattr(self.args, "eval_num_samples", len(detailed))
            ):
                metrics.update(
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
                metrics.update(extra_metrics)

        alias_sources = {
            "eval/reward": "eval/turn_1/reward_mean",
            "eval/reward_mean": "eval/turn_1/reward_mean",
            "eval/turn_1/reward": "eval/turn_1/reward_mean",
            "eval/action_validity": "eval/turn_1/action_validity",
            "eval/team_action_success": (
                "eval/turn_1/ultimate/team_action_success"
            ),
            "eval/both_agent_verified_contribution": (
                "eval/turn_1/ultimate/both_agent_verified_contribution"
            ),
            "eval/both_agent_required_grounded_contribution": (
                "eval/turn_1/ultimate/"
                "both_agent_required_grounded_contribution"
            ),
            "eval/collaboration_success": (
                "eval/turn_1/ultimate/collaboration_success"
            ),
            "eval/reference_plan_success": (
                "eval/turn_1/ultimate/reference_plan_success"
            ),
            "eval/reference_commonsense_micro": (
                "eval/turn_1/ultimate/reference_commonsense_micro"
            ),
            "eval/reference_hard_micro": (
                "eval/turn_1/ultimate/reference_hard_micro"
            ),
            "eval/required_grounded_recall": (
                "eval/turn_1/required_grounded_recall"
            ),
            "eval/required_cooperative_contribution": (
                "eval/turn_1/required_cooperative_contribution"
            ),
            "eval/entity_grounding_precision": (
                "eval/turn_1/entity_grounding_precision"
            ),
            "eval/grounding_f1": "eval/turn_1/grounding_f1",
            "eval/route_scaffold_match": (
                "eval/turn_1/route_scaffold_match_rate"
            ),
            "eval/plan_score": "eval/turn_1/reward_model/plan_score",
            "eval/strict_composite_quality": (
                "eval/turn_1/reward_model/strict_composite_quality"
            ),
            "eval/protocol_progress": (
                "eval/turn_1/reward_model/protocol_progress"
            ),
            "eval/recovered_semantic_balance": (
                "eval/turn_1/reward_model/recovered_semantic_balance"
            ),
            "eval/recovered_plan_score": (
                "eval/turn_1/reward_model/recovered_plan_score"
            ),
            "eval/recovered_composite_quality": (
                "eval/turn_1/reward_model/recovered_composite_quality"
            ),
            "eval_full/reward": "eval_full/turn_1/reward_mean",
            "eval_full/reward_mean": "eval_full/turn_1/reward_mean",
            "eval_full/collaboration_success": (
                "eval_full/turn_1/ultimate/collaboration_success"
            ),
            "eval_full/team_action_success": (
                "eval_full/turn_1/ultimate/team_action_success"
            ),
            "eval_full/required_cooperative_contribution": (
                "eval_full/turn_1/required_cooperative_contribution"
            ),
        }
        aliases = {
            alias: metrics[source]
            for alias, source in alias_sources.items()
            if source in metrics
        }
        metrics.update(aliases)
        headline_aliases = {
            "reward": "eval/reward",
            "team_action_success": "eval/team_action_success",
            "required_cooperative_contribution": (
                "eval/required_cooperative_contribution"
            ),
            "required_grounded_recall": "eval/required_grounded_recall",
            "entity_grounding_precision": "eval/entity_grounding_precision",
            "grounding_f1": "eval/grounding_f1",
            "reference_commonsense_micro": (
                "eval/reference_commonsense_micro"
            ),
            "reference_hard_micro": "eval/reference_hard_micro",
            "reference_plan_success": "eval/reference_plan_success",
            "collaboration_success": "eval/collaboration_success",
        }
        current_headlines = {
            label: float(metrics[key])
            for label, key in headline_aliases.items()
            if isinstance(metrics.get(key), Real)
        }
        if current_headlines and not getattr(self, "rotate_eval_subset", False):
            baseline = getattr(self, "_travel_eval_baseline", None)
            if baseline is None:
                baseline = dict(current_headlines)
                self._travel_eval_baseline = baseline
            comparable = sorted(set(baseline) & set(current_headlines))
            for label in comparable:
                metrics[f"eval/delta/{label}"] = (
                    current_headlines[label] - baseline[label]
                )
            table_factory = getattr(wandb, "Table", None)
            if (
                self.wandb_initialized
                and wandb.run is not None
                and callable(table_factory)
            ):
                metrics["eval/headline_summary"] = table_factory(
                    columns=["metric", "initial", "current", "delta"],
                    data=[
                        [
                            label,
                            baseline[label],
                            current_headlines[label],
                            current_headlines[label] - baseline[label],
                        ]
                        for label in comparable
                    ],
                )
        if self.wandb_initialized and wandb.run is not None:
            # Log the table, detailed metrics, and aliases in one call. W&B
            # discards a second committed log at the same explicit step.
            wandb.log(metrics, step=self.env_step, commit=True)
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
        if (
            getattr(self, "_travel_eval_generation", False)
            and getattr(self, "greedy_eval", True)
        ):
            # Keep the fixed anchor comparable within a run without changing
            # stock MAGRPO or disabling stochastic training rollouts.
            kwargs["logits_processor"] = with_greedy_argmax_processor(
                kwargs.get("logits_processor")
            )
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

        result["prompts"] = prompts
        result["completions"] = [completion_texts]
        result["completion_input_ids"] = [cropped_tokens]
        result["completion_attention_mask"] = [completion_masks]
        result["response_lens"] = response_lens
        return result
