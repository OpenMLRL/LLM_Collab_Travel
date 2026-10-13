"""One jointly planned itinerary, projected to the unchanged two actor roles."""

from __future__ import annotations

import json

import torch

from single_turn.aggregation import PLAN_FIELDS, ordered_owned_slots
from single_turn.formatting import build_agent_json_prefill, get_single_turn_formatters


def joint_slots(days):
    return [(day, field) for day in range(1, int(days) + 1) for field in PLAN_FIELDS]


class TravelCentralizedComparatorAdapter:
    """Centralization changes information sharing, not roles, reward or merge."""

    def build_prompt(self, batch_item, agent_prompts):
        del agent_prompts  # Rebuild the same role content without nested chat tokens.
        raw = get_single_turn_formatters(num_agents=2, use_chat_template=False)
        sections = "\n\n".join(
            f"REFERENCE ROLE {i} (context only):\n{formatter(batch_item)}"
            for i, formatter in enumerate(raw)
        )
        slots = joint_slots(batch_item["days"])
        return f"""You are ONE centralized travel planner, not one of two isolated agents.
You see both roles' information and choose the entire itinerary jointly in one response.
The reference role instructions below specify the same trip, catalogs, constraints,
budget guidance and downstream responsibilities used by the two-agent team.
Their statements about acting independently and their per-role output envelopes
do NOT apply to you. You may coordinate all logistics and experiences directly.
Use no information beyond these role contexts. Do not use a reference/gold itinerary.

{sections}

YOUR CENTRALIZED OUTPUT CONTRACT (overrides the reference role output contracts):
Return ONE JSON object with agent_id=0 and exactly {len(slots)} assignments.
Here agent_id=0 labels the central carrier, not a restriction to logistics.
Choose EVERY field of EVERY day, in this day-major order:
{json.dumps(slots, ensure_ascii=False)}
The decoder supplies the same fixed JSON schema as the role decoders, with the
same per-value token limit. Choose only values. Copy catalog spelling exactly;
use "-" only where the task permits an intentionally empty field.
After generation, a deterministic splitter sends logistics to agent 0 and all
experiences (including dinner) to agent 1. The existing merger/reward is unchanged.
The assistant prefill is already supplied. Continue inside its first value string.
"""

    def parse_completion(self, completion, batch_item, num_agents):
        if num_agents != 2:
            raise ValueError("Travel centralized comparator requires two actor roles.")
        data = json.loads(completion)
        if (
            not isinstance(data, dict)
            or type(data.get("agent_id")) is not int
            or data["agent_id"] != 0
        ):
            raise ValueError("Centralized Travel output must have integer agent_id=0.")
        assignments = data.get("assignments")
        expected = joint_slots(batch_item["days"])
        if not isinstance(assignments, list) or len(assignments) != len(expected):
            raise ValueError(
                "Centralized Travel output must contain every joint slot exactly once."
            )
        values = {}
        for assignment, slot in zip(assignments, expected):
            if not isinstance(assignment, dict) or set(assignment) != {
                "day",
                "field",
                "value",
            }:
                raise ValueError("Invalid centralized Travel assignment schema.")
            if (
                type(assignment["day"]) is not int
                or (assignment["day"], assignment["field"]) != slot
            ):
                raise ValueError(
                    "Centralized Travel slots must be in the joint day-major order."
                )
            if not isinstance(assignment["value"], str):
                raise ValueError("Centralized Travel values must be strings.")
            values[slot] = assignment["value"]
        return [
            json.dumps(
                {
                    "agent_id": i,
                    "assignments": [
                        {"day": day, "field": field, "value": values[day, field]}
                        for day, field in ordered_owned_slots(i, batch_item["days"])
                    ],
                },
                ensure_ascii=False,
            )
            for i in range(2)
        ]


def project_role_tokens(tokenizer, completion, agent_idx, days):
    """Serialize a central teacher's role projection under the actor's grammar.

    Central candidates were sampled under a joint prompt, so their joint token
    stream cannot be reused as an actor continuation. Encode value/closing-quote
    and forced-schema pieces separately, excluding the assistant prefill. Native
    decentralized trajectories continue to retain their exact sampled tokens.
    """
    data = json.loads(completion)
    slots = ordered_owned_slots(agent_idx, days)
    assignments = data["assignments"]
    if data["agent_id"] != agent_idx or len(assignments) != len(slots):
        raise ValueError("Central comparator projection does not match the actor role.")
    tokens, mask, pieces = [], [], []
    for index, ((day, field), assignment) in enumerate(zip(slots, assignments)):
        if (assignment["day"], assignment["field"]) != (day, field) or not isinstance(
            assignment["value"], str
        ):
            raise ValueError(
                "Central comparator projection has invalid role slots/values."
            )
        value = json.dumps(assignment["value"], ensure_ascii=False)[1:]
        if index + 1 == len(slots):
            suffix = "}]}"
        else:
            next_day, next_field = slots[index + 1]
            suffix = f'}}, {{"day": {next_day}, "field": {json.dumps(next_field)}, "value": "'
        for text, learnable in ((value, True), (suffix, False)):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if (
                tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                != text
            ):
                raise ValueError(
                    "Tokenizer does not round-trip the projected Travel action."
                )
            tokens.extend(ids)
            mask.extend([learnable] * len(ids))
            pieces.append(text)
    text = build_agent_json_prefill(agent_idx, days) + "".join(pieces)
    if json.loads(text) != data:
        raise ValueError("Centralized role projection changed assignment values.")
    return (
        text,
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )
