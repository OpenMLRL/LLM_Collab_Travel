"""Thin Travel structured-generation adapter for the stock MAGRPO trainer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from comlrl.trainers.reinforce import MAGRPOTrainer

from single_turn.structured_generation import (
    CompleteJSONObjectCriteria,
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
        **kwargs: Any,
    ):
        self.chat_formatted_prompts = bool(chat_formatted_prompts)
        self.force_json_prefix = bool(force_json_prefix)
        self.stop_after_complete_json = bool(stop_after_complete_json)
        self.rotate_eval_subset = bool(rotate_eval_subset)
        self._eval_cursor = 0
        super().__init__(*args, **kwargs)

    def evaluate(self, num_eval_samples: Optional[int] = None) -> Dict[str, float]:
        """Evaluate a cyclic held-out shard while leaving stock MAGRPO untouched."""

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
        if self.force_json_prefix:
            if not self.chat_formatted_prompts:
                raise ValueError(
                    "travel.force_json_prefix=true requires chat-formatted prompts."
                )
            # Treat the fixed opening brace as assistant prefill. It is therefore
            # context rather than a sampled policy token; the reward-facing text
            # is reconstructed after generation.
            prompts = [prompt + "{" for prompt in prompts]

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
                initial_text="{" if self.force_json_prefix else "",
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
                text = "{" + text
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
