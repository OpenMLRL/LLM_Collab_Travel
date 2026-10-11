"""Direct-only grammar accepting Qwen's fused value/JSON-boundary tokens.

The team grammar admits a quote only at a token's end. Qwen3-8B frequently
predicts a token such as '\"},'; masking that token made it substitute apostrophe
delimiters *inside* entity values. Accept those real boundary tokens and force
only the still-unconsumed schema suffix. Existing team generation is untouched.
"""

from functools import lru_cache
from threading import Lock
import torch
from single_turn.aggregation import PLAN_FIELDS
from single_turn.structured_generation import FixedSlotJSONLogitsProcessor

_DIRECT_MASK_CACHE = {}
_DIRECT_MASK_LOCK = Lock()
_SUFFIXES = ("}]}",) + tuple(
    f'}}, {{"day": {day}, "field": "{field}", "value": "'
    for day in range(1, 8)
    for field in PLAN_FIELDS
)


@lru_cache(maxsize=1000000)
def fragment_parts(text, initial_escape):
    escape = initial_escape
    for i, char in enumerate(text):
        if escape == 1:
            escape = 2 if char == "u" else 0
        elif escape >= 2:
            escape = 0 if escape == 5 else escape + 1
        elif char == "\\":
            escape = 1
        elif char == '"':
            kind, state, content = FixedSlotJSONLogitsProcessor._advance_value_fragment(
                text[: i + 1], initial_escape
            )
            tail = text[i + 1 :]
            if kind != "close" or not any(s.startswith(tail) for s in _SUFFIXES):
                return "invalid", state, content, tail
            return kind, state, content, tail
    kind, state, content = FixedSlotJSONLogitsProcessor._advance_value_fragment(
        text, initial_escape
    )
    return kind, state, content, ""


class DirectJSONGrammar(FixedSlotJSONLogitsProcessor):
    @staticmethod
    def _advance_value_fragment(text, escape_state):
        return fragment_parts(text, escape_state)[:3]

    def _consume_token(self, state, token_id):
        if state.mode != "value":
            return super()._consume_token(state, token_id)
        fragment = self.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        kind, escape, content, tail = fragment_parts(fragment, state.escape_state)
        if kind != "close":
            return super()._consume_token(state, token_id)
        suffix = self._suffix_after(state.slot_index)
        if not suffix.startswith(tail) or not (state.value_has_content or content):
            raise RuntimeError("Direct grammar accepted the wrong boundary token")
        state.loss_mask.append(
            1
        )  # One sampled token includes value/voluntary boundary.
        state.generated_token_count += 1
        state.value_token_count += 1
        state.value_has_content |= content
        state.escape_state = escape
        rest = suffix[len(tail) :]
        if rest:
            self._schedule(state, "skeleton", self._encode_exact(rest))
        else:
            self._finish_pending(state, "skeleton")

    def __call__(self, input_ids, scores):
        constrained = super().__call__(input_ids, scores)
        # A closing token may contain a suffix prefix. The final slot's suffix
        # differs from the others: forbid a continuation belonging to another slot.
        if not hasattr(self, "_boundary_masks"):
            self._boundary_masks = {}
        for i, state in enumerate(self._states):
            if state.mode != "value":
                continue
            suffix = self._suffix_after(state.slot_index)
            key = (state.escape_state, suffix, str(scores.device))
            if key not in self._boundary_masks:
                ids = [
                    j
                    for j, tail in self._tail_ids[state.escape_state]
                    if not suffix.startswith(tail)
                ]
                self._boundary_masks[key] = torch.tensor(
                    ids, device=scores.device, dtype=torch.long
                )
            constrained[i, self._boundary_masks[key]] = -torch.inf
        return constrained

    def _prepare_vocab(self, vocab_size: int) -> None:
        if (
            self._value_masks_cpu
            and next(iter(self._value_masks_cpu.values())).numel() == vocab_size
        ):
            return
        # Direct rollouts share one tokenizer object and can start generation
        # on separate threads. Build its O(V) masks only once.
        with _DIRECT_MASK_LOCK:
            cached = _DIRECT_MASK_CACHE.get(id(self.tokenizer))
            if (
                cached is not None
                and cached[0] is self.tokenizer
                and cached[1] == vocab_size
            ):
                _tokenizer, _vocab_size, self._value_masks_cpu, self._tail_ids = cached
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
                for token_id in (getattr(self.tokenizer, "all_special_ids", None) or [])
                if 0 <= int(token_id) < vocab_size
            }
            fragments = [None] * vocab_size
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
            tail_ids = {escape: [] for escape in range(6)}
            for escape_state in range(6):
                for token_id, fragment in enumerate(fragments):
                    if fragment is None:
                        continue
                    kind, _next_escape, token_has_content = (
                        self._advance_value_fragment(fragment, escape_state)
                    )
                    if kind == "close":
                        tail_ids[escape_state].append(
                            (token_id, fragment_parts(fragment, escape_state)[3])
                        )
                    if kind == "invalid":
                        continue
                    for value_has_content in (False, True):
                        if kind != "close" or value_has_content or token_has_content:
                            masks[(escape_state, value_has_content)][token_id] = True
            if not masks[(0, False)].any() or not any(
                self._advance_value_fragment(fragment or "", 0)[0] == "close"
                for fragment in fragments
            ):
                raise ValueError(
                    "Tokenizer vocabulary cannot express constrained JSON values."
                )
            self._value_masks_cpu = masks
            self._tail_ids = tail_ids
            _DIRECT_MASK_CACHE[id(self.tokenizer)] = (
                self.tokenizer,
                vocab_size,
                masks,
                tail_ids,
            )
            self._mask_cache.clear()
