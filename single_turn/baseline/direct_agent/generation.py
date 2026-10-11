"""One complete itinerary with Travel's existing fixed-slot action grammar."""

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from single_turn.structured_generation import CompleteJSONObjectCriteria
from .grammar import DirectJSONGrammar
from .protocol import PREFILL, slots


def generate(model, tokenizer, prompt, count, cfg, *, row):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    samples = []
    try:
        for offset in range(0, count, cfg.generation_batch_size):
            n = min(cfg.generation_batch_size, count - offset)
            grammar = DirectJSONGrammar(
                tokenizer,
                prompt_length=prompt.numel(),
                slots=slots(int(row["days"])),
                max_value_tokens=cfg.max_value_tokens,
                max_new_tokens=cfg.max_new_tokens,
            )
            stop = CompleteJSONObjectCriteria(
                tokenizer, prompt_length=prompt.numel(), initial_text=PREFILL
            )
            ids = prompt.to(device)[None]
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    num_return_sequences=n,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([grammar]),
                    stopping_criteria=StoppingCriteriaList([stop]),
                )
            tokens = []
            for i, output in enumerate(outputs):
                length = stop.completed_response_lengths[i]
                if length is None:
                    raise RuntimeError(
                        "Fixed Travel JSON did not complete within the token budget"
                    )
                tokens.append(
                    output[prompt.numel() : prompt.numel() + length]
                    .detach()
                    .cpu()
                    .clone()
                )
            masks = grammar.finalize_loss_masks(tokens)
            for token_row, mask in zip(tokens, masks):
                mask = torch.tensor(mask, dtype=torch.bool)
                if mask.numel() != token_row.numel() or not mask.any():
                    raise RuntimeError("No policy-selected Travel value tokens")
                text = PREFILL + tokenizer.decode(
                    token_row,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                samples.append((token_row, text, mask))
    finally:
        model.train(was_training)
    return samples


class _PerRowGrammar(LogitsProcessor):
    def __init__(self, grammars):
        self.grammars = grammars

    def __call__(self, input_ids, scores):
        return torch.cat(
            [
                grammar(input_ids[i : i + 1], scores[i : i + 1])
                for i, grammar in enumerate(self.grammars)
            ],
            dim=0,
        )


class _PerRowStop(StoppingCriteria):
    def __init__(self, stops):
        self.stops = stops

    def __call__(self, input_ids, scores, **kwargs):
        return torch.cat(
            [stop(input_ids[i : i + 1], scores) for i, stop in enumerate(self.stops)],
            dim=0,
        )


def generate_prompt_batch(model, tokenizer, prompts, cfg, *, rows):
    """One independent response per question, concurrently for the IAC batch.

    Left padding/attention masks preserve each question's context. No example
    sees another question; optimizer grouping and one-sample IAC are unchanged.
    """
    if not prompts or len(prompts) != len(rows):
        raise ValueError("Expected matching nonempty prompts and rows")
    device = next(model.parameters()).device
    width = max(p.numel() for p in prompts)
    ids = torch.full(
        (len(prompts), width), tokenizer.pad_token_id, device=device, dtype=torch.long
    )
    attention = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, -prompt.numel() :] = prompt.to(device)
        attention[i, -prompt.numel() :] = 1
    grammars = [
        DirectJSONGrammar(
            tokenizer,
            prompt_length=width,
            slots=slots(int(row["days"])),
            max_value_tokens=cfg.max_value_tokens,
            max_new_tokens=cfg.max_new_tokens,
        )
        for row in rows
    ]
    stops = [
        CompleteJSONObjectCriteria(tokenizer, prompt_length=width, initial_text=PREFILL)
        for _ in rows
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=ids,
                attention_mask=attention,
                num_return_sequences=1,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                logits_processor=LogitsProcessorList([_PerRowGrammar(grammars)]),
                stopping_criteria=StoppingCriteriaList([_PerRowStop(stops)]),
            )
        result = []
        for i, (grammar, stop) in enumerate(zip(grammars, stops)):
            length = stop.completed_response_lengths[0]
            if length is None:
                raise RuntimeError("Batched direct itinerary did not complete")
            tokens = outputs[i, width : width + length].detach().cpu().clone()
            mask = torch.tensor(
                grammar.finalize_loss_masks([tokens])[0], dtype=torch.bool
            )
            if tokens.numel() != mask.numel() or not mask.any():
                raise RuntimeError("Invalid batched value-token mask")
            text = PREFILL + tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result.append((tokens, text, mask))
        return result
    finally:
        model.train(was_training)
