"""Utilities for chat-formatted, single-JSON model generations.

The strict Travel reward intentionally rejects prefixes, suffixes, and additional
JSON objects.  These helpers make that output contract part of generation rather
than weakening the parser after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)


DEFAULT_SYSTEM_PROMPT = (
    "You are one agent in a decentralized travel-planning team. Follow the user's "
    "role and output contract exactly. Return only the requested JSON object."
)


# Vocabulary classification is tokenizer-specific but independent of a sample.
# A training run creates one grammar processor per rollout, so cache this O(V)
# scan instead of decoding the full vocabulary for every Travel prompt.
_VALUE_TOKEN_MASK_CACHE: Dict[int, Tuple[Any, int, Dict[Tuple[int, bool], Any]]] = {}
_VALUE_TOKEN_MASK_CACHE_LOCK = Lock()


class GreedyArgmaxLogitsProcessor(LogitsProcessor):
    """Leave exactly one next-token candidate for sampling-free eval.

    MAGRPO intentionally samples during rollout generation.  Travel reuses that
    path for evaluation, so this processor makes the evaluation path equivalent
    to greedy argmax without changing the shared trainer or the stochastic
    training path.  ``argmax`` also resolves exact logit ties deterministically
    by selecting the first token index.
    """

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        del input_ids
        winner = scores.argmax(dim=-1, keepdim=True)
        forced = torch.full_like(scores, float("-inf"))
        return forced.scatter(1, winner, scores.gather(1, winner))


def with_greedy_argmax_processor(existing: Any) -> LogitsProcessorList:
    """Append Travel's argmax constraint without mutating caller processors."""

    processor = GreedyArgmaxLogitsProcessor()
    if existing is None:
        return LogitsProcessorList([processor])
    if isinstance(existing, LogitsProcessorList):
        return LogitsProcessorList([*existing, processor])
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        return LogitsProcessorList([*existing, processor])
    return LogitsProcessorList([existing, processor])


def with_fixed_slot_json_processor(
    existing: Any, processor: "FixedSlotJSONLogitsProcessor"
) -> LogitsProcessorList:
    """Append a prepared fixed-slot processor without mutating caller state."""

    if existing is None:
        return LogitsProcessorList([processor])
    if isinstance(existing, LogitsProcessorList):
        return LogitsProcessorList([*existing, processor])
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        return LogitsProcessorList([*existing, processor])
    return LogitsProcessorList([existing, processor])


@dataclass
class _FixedSlotState:
    """Per-sequence state for the fixed Travel assignment skeleton."""

    slot_index: int = 0
    mode: str = "value"
    pending_tokens: Tuple[int, ...] = ()
    pending_index: int = 0
    value_token_count: int = 0
    value_has_content: bool = False
    escape_state: int = 0
    generated_token_count: int = 0
    loss_mask: List[int] = field(default_factory=list)
    invalid: bool = False


class FixedSlotJSONLogitsProcessor(LogitsProcessor):
    """Let the policy choose values while forcing the role-owned JSON skeleton.

    The assistant prompt already contains the JSON prefix through the first
    assignment's opening value quote.  While a value is open, this processor
    admits ordinary JSON-string text and a closing quote.  Once the policy
    closes the value, every key, day, field, comma, bracket, and brace up to the
    next value is forced.  This removes syntax and slot-order decisions from the
    Travel action space without changing the shared MAGRPO implementation.

    A loss mask records which tokens were selected by the policy (value text and
    a voluntary closing quote) versus forced schema tokens.  The Travel trainer
    uses that mask so deterministic scaffolding cannot dominate the policy loss.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_length: int,
        slots: Sequence[Tuple[int, str]],
        max_value_tokens: int = 32,
        max_new_tokens: Optional[int] = None,
    ):
        if int(prompt_length) < 0:
            raise ValueError("prompt_length must be non-negative.")
        if not slots:
            raise ValueError("Fixed-slot JSON generation requires at least one slot.")
        if int(max_value_tokens) < 1:
            raise ValueError("max_value_tokens must be positive.")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.slots = tuple((int(day), str(field)) for day, field in slots)
        self.max_value_tokens = int(max_value_tokens)
        self.max_new_tokens = (
            None if max_new_tokens is None else int(max_new_tokens)
        )
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive when supplied.")

        self._quote_tokens = self._encode_exact('"')
        self._fallback_close_tokens = self._encode_exact('-"')
        self._escape_close_tokens = {
            0: self._quote_tokens,
            1: self._encode_exact('n"'),
            2: self._encode_exact('0000"'),
            3: self._encode_exact('000"'),
            4: self._encode_exact('00"'),
            5: self._encode_exact('0"'),
        }
        self._max_forced_close_length = max(
            len(self._fallback_close_tokens),
            *(len(tokens) for tokens in self._escape_close_tokens.values()),
        )
        self._suffix_tokens = tuple(
            self._encode_exact(self._suffix_after(slot_index))
            for slot_index in range(len(self.slots))
        )
        blank_tokens: Tuple[int, ...] = ()
        blank_text = ""
        for slot_index, suffix_tokens in enumerate(self._suffix_tokens):
            blank_tokens += self._fallback_close_tokens + suffix_tokens
            blank_text += '-"' + self._suffix_after(slot_index)
        decoded_blank = self.tokenizer.decode(
            list(blank_tokens),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded_blank != blank_text:
            raise ValueError(
                "Tokenizer cannot compositionally decode the fixed Travel JSON "
                "skeleton."
            )
        self._states: List[_FixedSlotState] = []
        self._value_masks_cpu: Dict[Tuple[int, bool], torch.BoolTensor] = {}
        self._mask_cache: Dict[Tuple[str, int, bool], torch.BoolTensor] = {}

        minimum = self._minimum_blank_response_tokens(0)
        if self.max_new_tokens is not None and minimum > self.max_new_tokens:
            raise ValueError(
                "max_new_tokens cannot fit the fixed Travel JSON skeleton: "
                f"need at least {minimum}, got {self.max_new_tokens}."
            )

    def _encode_exact(self, text: str) -> Tuple[int, ...]:
        token_ids = tuple(
            int(token_id)
            for token_id in self.tokenizer.encode(text, add_special_tokens=False)
        )
        if not token_ids:
            raise ValueError(f"Tokenizer cannot encode required JSON text {text!r}.")
        decoded = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            raise ValueError(
                "Tokenizer does not round-trip required JSON text: "
                f"expected {text!r}, decoded {decoded!r}."
            )
        return token_ids

    def _suffix_after(self, slot_index: int) -> str:
        if slot_index == len(self.slots) - 1:
            return "}]}"
        day, field = self.slots[slot_index + 1]
        return (
            f'}}, {{"day": {day}, "field": '
            f'{json.dumps(field, ensure_ascii=False)}, "value": "'
        )

    @staticmethod
    def _advance_value_fragment(
        text: str, escape_state: int
    ) -> Tuple[str, int, bool]:
        """Advance a JSON-string DFA over one decoded vocabulary token.

        Escape state 0 is ordinary text, 1 follows a backslash, and 2..5
        represent four through one remaining hexadecimal digits in a ``\\u``
        escape. A closing quote is admitted only as the token's final character,
        ensuring that all following assignment syntax comes from the fixed
        skeleton. Travel catalog entries containing quotes or newlines therefore
        remain expressible as normal JSON escape sequences.
        """

        if not text or escape_state not in range(6):
            return "invalid", escape_state, False
        state = int(escape_state)
        has_content = False
        for char_index, char in enumerate(text):
            if state == 1:
                if char in '"\\/bfnrt':
                    state = 0
                    has_content = True
                    continue
                if char == "u":
                    state = 2
                    has_content = True
                    continue
                return "invalid", state, has_content
            if state >= 2:
                if char not in "0123456789abcdefABCDEF":
                    return "invalid", state, has_content
                state = 0 if state == 5 else state + 1
                has_content = True
                continue
            if ord(char) < 0x20:
                return "invalid", state, has_content
            if char == "\\":
                state = 1
                has_content = True
                continue
            if char == '"':
                if char_index != len(text) - 1:
                    return "invalid", state, has_content
                return "close", 0, has_content
            has_content = has_content or bool(char.strip())
        return "continue", state, has_content

    def _prepare_vocab(self, vocab_size: int) -> None:
        if self._value_masks_cpu and next(
            iter(self._value_masks_cpu.values())
        ).numel() == vocab_size:
            return
        # Both Travel agents share one tokenizer object and can start generation
        # on separate threads. Build its O(V) masks only once.
        with _VALUE_TOKEN_MASK_CACHE_LOCK:
            cached = _VALUE_TOKEN_MASK_CACHE.get(id(self.tokenizer))
            if (
                cached is not None
                and cached[0] is self.tokenizer
                and cached[1] == vocab_size
            ):
                _tokenizer, _vocab_size, self._value_masks_cpu = cached
                self._mask_cache.clear()
                return
            masks = {
                (escape_state, value_has_content): torch.zeros(
                    vocab_size, dtype=torch.bool
                )
                for escape_state in range(6)
                for value_has_content in (False, True)
            }
            special_ids = {
                int(token_id)
                for token_id in (
                    getattr(self.tokenizer, "all_special_ids", None) or []
                )
                if 0 <= int(token_id) < vocab_size
            }
            fragments: List[Optional[str]] = [None] * vocab_size
            for token_id in range(vocab_size):
                if token_id in special_ids:
                    continue
                try:
                    fragments[token_id] = self.tokenizer.decode(
                        [token_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                except Exception:
                    continue
            for escape_state in range(6):
                for token_id, fragment in enumerate(fragments):
                    if fragment is None:
                        continue
                    kind, _next_escape, token_has_content = (
                        self._advance_value_fragment(fragment, escape_state)
                    )
                    if kind == "invalid":
                        continue
                    for value_has_content in (False, True):
                        if (
                            kind != "close"
                            or value_has_content
                            or token_has_content
                        ):
                            masks[(escape_state, value_has_content)][token_id] = True
            if not masks[(0, False)].any() or not any(
                self._advance_value_fragment(fragment or "", 0)[0] == "close"
                for fragment in fragments
            ):
                raise ValueError(
                    "Tokenizer vocabulary cannot express constrained JSON values."
                )
            self._value_masks_cpu = masks
            _VALUE_TOKEN_MASK_CACHE[id(self.tokenizer)] = (
                self.tokenizer,
                vocab_size,
                masks,
            )
            self._mask_cache.clear()

    def _allowed_value_mask(
        self,
        device: torch.device,
        *,
        escape_state: int,
        value_has_content: bool,
    ) -> torch.BoolTensor:
        key = (str(device), int(escape_state), bool(value_has_content))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = self._value_masks_cpu[
            (int(escape_state), bool(value_has_content))
        ].to(
            device=device
        )
        self._mask_cache[key] = mask
        return mask

    def _reset(self, batch_size: int) -> None:
        self._states = [_FixedSlotState() for _ in range(batch_size)]

    def _schedule(self, state: _FixedSlotState, mode: str, tokens: Sequence[int]) -> None:
        if not tokens:
            raise RuntimeError("Cannot schedule an empty constrained token sequence.")
        state.mode = mode
        state.pending_tokens = tuple(int(token_id) for token_id in tokens)
        state.pending_index = 0

    def _schedule_suffix(self, state: _FixedSlotState) -> None:
        self._schedule(state, "skeleton", self._suffix_tokens[state.slot_index])

    def _forced_value_close_tokens(
        self, state: _FixedSlotState
    ) -> Tuple[int, ...]:
        """Return tokens that finish any partial escape and close the value."""

        if not state.value_has_content and state.escape_state == 0:
            return self._fallback_close_tokens
        return self._escape_close_tokens[state.escape_state]

    def _finish_pending(self, state: _FixedSlotState, completed_mode: str) -> None:
        state.pending_tokens = ()
        state.pending_index = 0
        if completed_mode in {"forced_close", "forced_fallback"}:
            self._schedule_suffix(state)
            return
        if state.slot_index == len(self.slots) - 1:
            state.mode = "complete"
            return
        state.slot_index += 1
        state.mode = "value"
        state.value_token_count = 0
        state.value_has_content = False
        state.escape_state = 0

    def _consume_token(self, state: _FixedSlotState, token_id: int) -> None:
        token_id = int(token_id)
        if state.mode == "value":
            fragment = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            kind, next_escape_state, has_content = self._advance_value_fragment(
                fragment, state.escape_state
            )
            if kind == "invalid" or (
                kind == "close"
                and not (state.value_has_content or has_content)
            ):
                state.invalid = True
                raise RuntimeError(
                    "Fixed-slot JSON processor admitted an invalid value token."
                )
            state.loss_mask.append(1)
            state.generated_token_count += 1
            state.value_has_content = state.value_has_content or has_content
            state.escape_state = next_escape_state
            state.value_token_count += 1
            if kind == "close":
                self._schedule_suffix(state)
            return

        if state.mode in {"forced_close", "forced_fallback", "skeleton"}:
            expected = state.pending_tokens[state.pending_index]
            if token_id != expected:
                state.invalid = True
                raise RuntimeError(
                    "A generation processor overrode the fixed Travel JSON skeleton."
                )
            completed_mode = state.mode
            state.loss_mask.append(0)
            state.generated_token_count += 1
            state.pending_index += 1
            if state.pending_index == len(state.pending_tokens):
                self._finish_pending(state, completed_mode)
            return

        # Finished rows can receive padding/EOS while other return sequences
        # finish. Those tokens are cropped by CompleteJSONObjectCriteria.
        state.loss_mask.append(0)
        state.generated_token_count += 1

    def _consume_new_tokens(self, input_ids: torch.LongTensor) -> None:
        batch_size, sequence_length = input_ids.shape
        if len(self._states) != batch_size or any(
            self.prompt_length + state.generated_token_count > sequence_length
            for state in self._states
        ):
            self._reset(batch_size)
        starts = [
            self.prompt_length + state.generated_token_count
            for state in self._states
        ]
        earliest = min(starts, default=sequence_length)
        # Copy the newly generated batch tail from CUDA only once. During
        # ordinary generation every row has exactly one new token here.
        new_rows = input_ids[:, earliest:sequence_length].tolist()
        for row_idx, (state, start) in enumerate(zip(self._states, starts)):
            offset = start - earliest
            for token_id in new_rows[row_idx][offset:]:
                self._consume_token(state, int(token_id))

    def _minimum_blank_response_tokens(self, start_slot: int) -> int:
        return sum(
            len(self._fallback_close_tokens) + len(self._suffix_tokens[slot_index])
            for slot_index in range(start_slot, len(self.slots))
        )

    def _minimum_finish_tokens(self, state: _FixedSlotState) -> int:
        close_count = len(self._forced_value_close_tokens(state))
        return (
            close_count
            + len(self._suffix_tokens[state.slot_index])
            + self._minimum_blank_response_tokens(state.slot_index + 1)
        )

    def _must_close_value(self, state: _FixedSlotState) -> bool:
        if state.value_token_count >= self.max_value_tokens:
            return True
        if self.max_new_tokens is None:
            return False
        remaining = self.max_new_tokens - state.generated_token_count
        if remaining <= self._minimum_finish_tokens(state):
            return True
        # A newly sampled token may end in ``\`` or a partial ``\uXXXX``
        # escape, making its forced close longer than the current state's.
        # Only leave the value open when even that worst next state can still
        # be completed inside the generation budget.
        fixed_tail = (
            len(self._suffix_tokens[state.slot_index])
            + self._minimum_blank_response_tokens(state.slot_index + 1)
        )
        safe_choice_budget = 1 + self._max_forced_close_length + fixed_tail
        return remaining < safe_choice_budget

    @staticmethod
    def _force_token(
        scores: torch.FloatTensor, row_idx: int, token_id: int
    ) -> None:
        scores[row_idx].fill_(float("-inf"))
        scores[row_idx, int(token_id)] = 0.0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self._prepare_vocab(int(scores.shape[-1]))
        self._consume_new_tokens(input_ids)
        constrained = scores.clone()
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        if eos_id is None:
            eos_id = getattr(self.tokenizer, "pad_token_id", None)
        if eos_id is None:
            raise ValueError("Tokenizer requires an EOS or pad token for JSON generation.")

        for row_idx, state in enumerate(self._states):
            if state.mode == "value" and self._must_close_value(state):
                close_tokens = self._forced_value_close_tokens(state)
                self._schedule(
                    state,
                    (
                        "forced_close"
                        if state.value_has_content
                        else "forced_fallback"
                    ),
                    close_tokens,
                )

            if state.mode in {"forced_close", "forced_fallback", "skeleton"}:
                self._force_token(
                    constrained,
                    row_idx,
                    state.pending_tokens[state.pending_index],
                )
            elif state.mode == "complete":
                self._force_token(constrained, row_idx, int(eos_id))
            else:
                allowed = self._allowed_value_mask(
                    constrained.device,
                    escape_state=state.escape_state,
                    value_has_content=state.value_has_content,
                )
                constrained[row_idx].masked_fill_(~allowed, float("-inf"))
        return constrained

    def finalize_loss_masks(
        self, generated_rows: Sequence[torch.LongTensor]
    ) -> Tuple[Tuple[int, ...], ...]:
        """Consume each cropped final token and return policy-token masks."""

        if len(self._states) != len(generated_rows):
            self._reset(len(generated_rows))
        for state, token_row in zip(self._states, generated_rows):
            tokens = token_row.tolist()
            for token_id in tokens[state.generated_token_count :]:
                self._consume_token(state, int(token_id))
        masks = []
        for state, token_row in zip(self._states, generated_rows):
            masks.append(tuple(state.loss_mask[: int(token_row.numel())]))
        return tuple(masks)


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
    delimiters: List[str] = field(default_factory=list)
    in_string: bool = False
    escaped: bool = False
    invalid: bool = False
    complete: bool = False
    terminal: bool = False

    def feed(self, text: str) -> bool:
        if self.terminal:
            return True

        for char in text:
            if not self.started:
                if char == "{":
                    self.started = True
                    self.delimiters.append("{")
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
            elif char in "{[":
                self.delimiters.append(char)
            elif char in "}]":
                expected = "{" if char == "}" else "["
                if not self.delimiters or self.delimiters[-1] != expected:
                    # The raw action is already invalid JSON.  Do not crop it
                    # as a successful object merely because brace depth happens
                    # to reach zero while an assignments array is still open.
                    # Stop immediately anyway: extra tokens cannot repair a
                    # mismatched delimiter and would waste the rollout budget.
                    self.invalid = True
                    self.terminal = True
                    return True
                self.delimiters.pop()
                if not self.delimiters and not self.invalid:
                    self.complete = True
                    self.terminal = True
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
        active_rows = []
        starts = []
        for row_idx, state in enumerate(self._states):
            if state.terminal:
                done[row_idx] = True
                continue
            active_rows.append(row_idx)
            starts.append(
                max(self.prompt_length, self._processed_lengths[row_idx])
            )
        if not active_rows:
            return done

        earliest = min(starts)
        new_rows = input_ids[:, earliest:sequence_length].tolist()
        for row_idx, start in zip(active_rows, starts):
            state = self._states[row_idx]
            if sequence_length > start:
                offset = start - earliest
                new_token_ids = new_rows[row_idx][offset:]
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
            done[row_idx] = state.terminal
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
