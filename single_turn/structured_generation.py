"""Utilities for chat-formatted, single-JSON model generations.

The strict Travel reward intentionally rejects prefixes, suffixes, and additional
JSON objects.  These helpers make that output contract part of generation rather
than weakening the parser after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are one agent in a decentralized travel-planning team. Follow the user's "
    "role and output contract exactly. Return only the requested JSON object."
)


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    *,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render one Travel instruction as a model-native chat prompt.

    ``tokenize=False`` is deliberate: MAGRPO owns padding/truncation and later
    needs the rendered prompt as a string for rollout logging and reward calls.
    """

    template = getattr(tokenizer, "chat_template", None)
    renderer = getattr(tokenizer, "apply_chat_template", None)
    if not template or not callable(renderer):
        raise ValueError(
            "travel.use_chat_template=true requires a tokenizer with a chat_template. "
            "Set travel.use_chat_template=false only for a base model that genuinely "
            "expects raw text prompts."
        )

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(prompt)})
    rendered = renderer(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Tokenizer chat template returned an empty/non-string prompt.")
    return rendered


def wrap_formatter_with_chat_template(
    formatter: Callable[..., str],
    tokenizer: Any,
    *,
    enabled: bool = True,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> Callable[..., str]:
    """Wrap an existing dataset formatter without changing its call contract."""

    if not enabled:
        return formatter

    def chat_formatter(example: Dict[str, Any], external_prompts: Any = None) -> str:
        if external_prompts is None:
            prompt = formatter(example)
        else:
            prompt = formatter(example, external_prompts=external_prompts)
        return apply_chat_template(
            tokenizer,
            prompt,
            system_prompt=system_prompt,
        )

    return chat_formatter


@dataclass
class _JsonObjectState:
    """Incremental lexical state for the first JSON object in a text stream."""

    started: bool = False
    depth: int = 0
    in_string: bool = False
    escaped: bool = False
    complete: bool = False

    def feed(self, text: str) -> bool:
        if self.complete:
            return True

        for char in text:
            if not self.started:
                if char == "{":
                    self.started = True
                    self.depth = 1
                continue

            if self.in_string:
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                continue

            if char == '"':
                self.in_string = True
            elif char == "{":
                self.depth += 1
            elif char == "}":
                self.depth -= 1
                if self.depth == 0:
                    self.complete = True
                    return True

        return False


class CompleteJSONObjectCriteria(StoppingCriteria):
    """Stop each generated sequence after its first complete top-level JSON object.

    The prompt boundary is supplied per generation call.  This is important for
    Travel prompts because the instruction and reference data themselves contain
    many JSON objects.  State is incremental, avoiding repeated decoding/scanning
    of a potentially 1,024-token response on every generation step.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_length: int,
        initial_text: str = "",
    ):
        if int(prompt_length) < 0:
            raise ValueError("prompt_length must be non-negative.")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.initial_text = str(initial_text)
        self._states: List[_JsonObjectState] = []
        self._processed_lengths: List[int] = []
        self._completed_response_lengths: List[Optional[int]] = []

    def _reset(self, batch_size: int) -> None:
        self._states = [_JsonObjectState() for _ in range(batch_size)]
        if self.initial_text:
            for state in self._states:
                state.feed(self.initial_text)
        self._processed_lengths = [self.prompt_length for _ in range(batch_size)]
        self._completed_response_lengths = [None for _ in range(batch_size)]

    @property
    def completed_response_lengths(self) -> tuple[Optional[int], ...]:
        """Generated token counts at each row's first complete top-level object."""

        return tuple(self._completed_response_lengths)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        del scores, kwargs
        batch_size, sequence_length = input_ids.shape
        if len(self._states) != batch_size or any(
            processed > sequence_length for processed in self._processed_lengths
        ):
            self._reset(batch_size)

        done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        for row_idx in range(batch_size):
            state = self._states[row_idx]
            if state.complete:
                done[row_idx] = True
                continue

            start = max(self.prompt_length, self._processed_lengths[row_idx])
            if sequence_length > start:
                new_token_ids = input_ids[row_idx, start:sequence_length]
                fragment = self.tokenizer.decode(
                    new_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                completed_now = state.feed(fragment)
                self._processed_lengths[row_idx] = sequence_length
                if (
                    completed_now
                    and self._completed_response_lengths[row_idx] is None
                ):
                    self._completed_response_lengths[row_idx] = max(
                        0, sequence_length - self.prompt_length
                    )
            done[row_idx] = state.complete
        return done


def with_json_stopping_criterion(
    existing: Any,
    *,
    tokenizer: Any,
    prompt_length: int,
    criterion: Optional[CompleteJSONObjectCriteria] = None,
) -> StoppingCriteriaList:
    """Append Travel's JSON stop condition to optional caller criteria."""

    criterion = criterion or CompleteJSONObjectCriteria(
        tokenizer, prompt_length=prompt_length
    )
    if existing is None:
        return StoppingCriteriaList([criterion])
    if isinstance(existing, StoppingCriteriaList):
        return StoppingCriteriaList([*existing, criterion])
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        return StoppingCriteriaList([*existing, criterion])
    return StoppingCriteriaList([existing, criterion])
