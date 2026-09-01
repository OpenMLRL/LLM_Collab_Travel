"""Role-guided prompts for simultaneous TravelPlanner agents."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from single_turn.aggregation import PLAN_FIELDS, assignment_capacity


def _role_instruction(agent_idx: int, role_mode: str) -> str:
    if role_mode not in {"soft_roles", "same_prompt"}:
        raise ValueError(
            "role_mode must be 'soft_roles' or 'same_prompt' for the single-turn task."
        )
    if role_mode == "same_prompt":
        return (
            "Select a complementary subset of itinerary slots. You have no assigned "
            "specialty, so infer which contribution will most improve the merged plan."
        )

    if agent_idx == 0:
        core = (
            "Your primary role is LOGISTICS AND FEASIBILITY. Prioritize current_city, "
            "transportation, accommodation, route consistency, arrival/departure timing, "
            "and preserving budget for the other agent."
        )
        allowed = (
            "You may fill meal or attraction slots when useful; the role is guidance, "
            "not a hard action restriction."
        )
    elif agent_idx == 1:
        core = (
            "Your primary role is DAILY EXPERIENCE. Prioritize breakfast, attraction, "
            "lunch, dinner, user preferences, daily feasibility, and respecting the "
            "budget likely consumed by transport and accommodation."
        )
        allowed = (
            "You may fill city, transportation, or accommodation slots when useful; "
            "the role is guidance, not a hard action restriction."
        )
    else:
        core = f"You are agent {agent_idx + 1}; select useful uncovered itinerary slots."
        allowed = "The role is guidance, not a hard action restriction."

    return f"{core}\n{allowed}"


def build_single_turn_formatter(
    agent_idx: int,
    *,
    num_agents: int = 2,
    role_mode: str = "soft_roles",
) -> Callable[[Dict[str, Any]], str]:
    def formatter(example: Dict[str, Any], external_prompts: Any = None) -> str:
        del external_prompts
        days = int(example.get("days", 0))
        total_slots = len(PLAN_FIELDS) * days
        capacity = assignment_capacity(days, num_agents)
        reference = str(example.get("reference_information", ""))
        role = _role_instruction(agent_idx, role_mode)
        local_constraint = json.dumps(
            example.get("local_constraint", {}), ensure_ascii=False, sort_keys=True
        )
        fields = ", ".join(PLAN_FIELDS)
        return f"""You are Agent {agent_idx} in a decentralized travel-planning team.

There are {num_agents} agents. Every agent receives the same trip request and acts
simultaneously. You cannot communicate with the other agent or see its output. A
deterministic downstream merger combines all slot assignments into one itinerary.
Every agent receives exactly the same reward for that merged itinerary.

ROLE GUIDANCE:
{role}

COLLABORATION RULES:
- The itinerary has {total_slots} slots: {days} days x {len(PLAN_FIELDS)} fields.
- You may submit at most {capacity} assignments.
- Choose a useful complementary subset; do not attempt to submit the whole itinerary.
- A slot is identified by (day, field) and should be assigned exactly once by the team.
- Same-slot duplication wastes capacity; different values for one slot create a conflict.
- If a slot should intentionally be empty, explicitly assign the string "-". An omitted
  slot is considered missing, which is different from explicitly assigning "-".
- Use only entities and facts from REFERENCE INFORMATION.
- Do not invent prices, flight numbers, restaurants, attractions, or accommodations.
- Your own section is not scored separately; optimize the final team itinerary.

VALID FIELDS:
{fields}

TRIP REQUEST:
{example.get('query') or example.get('prompt') or ''}

STRUCTURED CONSTRAINTS:
origin={example.get('org', '')}
destination={example.get('dest', '')}
days={days}
dates={example.get('date', '')}
people={example.get('people_number', '')}
budget={example.get('budget', '')}
local_constraints={local_constraint}

REFERENCE INFORMATION:
{reference}

OUTPUT FORMAT:
Return one JSON object and nothing else:
{{
  "agent_id": {agent_idx},
  "assignments": [
    {{"day": 1, "field": "transportation", "value": "exact candidate text"}}
  ]
}}
"""

    return formatter


def get_single_turn_formatters(
    *,
    num_agents: int,
    role_mode: str = "soft_roles",
) -> List[Callable[[Dict[str, Any]], str]]:
    return [
        build_single_turn_formatter(
            agent_idx,
            num_agents=num_agents,
            role_mode=role_mode,
        )
        for agent_idx in range(num_agents)
    ]
