"""Strict action validation with best-effort assignment recovery."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Collection, List, Tuple


@dataclass(frozen=True)
class Assignment:
    day: Any
    field: Any
    value: Any


@dataclass(frozen=True)
class ParseResult:
    assignments: List[Assignment]
    parse_success: bool
    raw_item_count: int
    invalid_item_count: int
    error: str = ""
    decode_success: bool = False
    strict_json: bool = False
    schema_valid: bool = False
    agent_id: Any = None
    agent_id_match: bool = False
    capacity_valid: bool = True
    error_codes: Tuple[str, ...] = ()


def _strip_fences(text: str) -> str:
    stripped = (text or "").strip()
    if "```" not in stripped:
        return stripped
    blocks = re.findall(r"```(?:json|python|text)?\s*(.*?)```", stripped, re.S | re.I)
    return "\n".join(block.strip() for block in blocks) if blocks else stripped


def _recover_payload(text: str) -> Any:
    stripped = _strip_fences(text)
    if not stripped:
        raise ValueError("empty completion")

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(stripped)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("completion does not contain a JSON object or list")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_assignments(
    text: str,
    *,
    expected_agent_id: int | None = None,
    capacity: int | None = None,
    days: int | None = None,
    valid_fields: Collection[str] | None = None,
) -> ParseResult:
    """Parse an action while separating strict validity from recovered content.

    ``parse_success`` is true only for one complete JSON object with the exact
    schema, the expected agent id, valid item types, and no capacity overflow.
    When that contract fails, useful assignments are still recovered so the
    content reward remains dense early in training.
    """

    stripped = (text or "").strip()
    strict_payload: Any = None
    strict_json = False
    if stripped:
        try:
            strict_payload = json.loads(stripped)
            strict_json = True
        except (TypeError, json.JSONDecodeError):
            pass

    try:
        payload = strict_payload if strict_json else _recover_payload(text)
    except ValueError as exc:
        return ParseResult(
            [],
            False,
            0,
            0,
            str(exc),
            decode_success=False,
            strict_json=False,
            schema_valid=False,
            agent_id_match=False,
            capacity_valid=True,
            error_codes=("decode_failed",),
        )

    error_codes: List[str] = []
    if not strict_json:
        error_codes.append("not_strict_json")

    agent_id = payload.get("agent_id") if isinstance(payload, dict) else None
    top_level_valid = isinstance(payload, dict) and set(payload) == {
        "agent_id",
        "assignments",
    }
    if not top_level_valid:
        error_codes.append("top_level_schema")

    agent_id_valid = _is_int(agent_id)
    if expected_agent_id is None:
        agent_id_match = agent_id_valid
    else:
        agent_id_match = agent_id_valid and agent_id == int(expected_agent_id)
    if not agent_id_valid:
        error_codes.append("invalid_agent_id")
    elif not agent_id_match:
        error_codes.append("agent_id_mismatch")

    if isinstance(payload, dict):
        raw_items = payload.get("assignments", payload.get("slots", []))
    elif isinstance(payload, list):
        raw_items = payload
    else:
        return ParseResult(
            [],
            False,
            0,
            0,
            "top-level value must be an object or list",
            decode_success=True,
            strict_json=strict_json,
            schema_valid=False,
            agent_id=agent_id,
            agent_id_match=agent_id_match,
            capacity_valid=True,
            error_codes=tuple(error_codes + ["top_level_type"]),
        )

    if not isinstance(raw_items, list):
        return ParseResult(
            [],
            False,
            0,
            0,
            "assignments must be a list",
            decode_success=True,
            strict_json=strict_json,
            schema_valid=False,
            agent_id=agent_id,
            agent_id_match=agent_id_match,
            capacity_valid=True,
            error_codes=tuple(error_codes + ["assignments_type"]),
        )

    assignments: List[Assignment] = []
    invalid = 0
    strict_items_valid = True
    strict_values_valid = True
    strict_slots = set()
    has_self_duplicate = False
    valid_field_names = (
        {str(field) for field in valid_fields}
        if valid_fields is not None
        else None
    )
    for raw in raw_items:
        if isinstance(raw, dict) and {"field", "value"}.issubset(raw):
            assignments.append(
                Assignment(raw.get("day", raw.get("days")), raw["field"], raw["value"])
            )
            if (
                set(raw) != {"day", "field", "value"}
                or not _is_int(raw.get("day"))
                or not isinstance(raw.get("field"), str)
                or not isinstance(raw.get("value"), str)
            ):
                strict_items_valid = False
            else:
                normalized_field = raw["field"].strip().casefold()
                slot = (raw["day"], normalized_field)
                if slot in strict_slots:
                    has_self_duplicate = True
                strict_slots.add(slot)
                if days is not None and not 1 <= raw["day"] <= int(days):
                    strict_values_valid = False
                if (
                    valid_field_names is not None
                    and raw["field"] not in valid_field_names
                ):
                    strict_values_valid = False
                if not raw["value"].strip():
                    strict_values_valid = False
        elif isinstance(raw, (list, tuple)) and len(raw) == 3:
            assignments.append(Assignment(raw[0], raw[1], raw[2]))
            strict_items_valid = False
        else:
            invalid += 1
            strict_items_valid = False

    if not strict_items_valid:
        error_codes.append("assignment_schema")
    if not strict_values_valid:
        error_codes.append("assignment_value")
    if has_self_duplicate:
        error_codes.append("self_duplicate")
    capacity_valid = capacity is None or len(raw_items) <= int(capacity)
    if not capacity_valid:
        error_codes.append("capacity_exceeded")

    schema_valid = bool(top_level_valid and strict_items_valid and strict_values_valid)
    parse_success = bool(
        strict_json
        and schema_valid
        and agent_id_match
        and capacity_valid
        and not has_self_duplicate
    )

    return ParseResult(
        assignments=assignments,
        parse_success=parse_success,
        raw_item_count=len(raw_items),
        invalid_item_count=invalid,
        error=",".join(error_codes),
        decode_success=True,
        strict_json=strict_json,
        schema_valid=schema_valid,
        agent_id=agent_id,
        agent_id_match=agent_id_match,
        capacity_valid=capacity_valid,
        error_codes=tuple(error_codes),
    )
