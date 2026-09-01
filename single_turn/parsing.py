"""Robust parsing for partial TravelPlanner slot assignments."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, List


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


def _strip_fences(text: str) -> str:
    stripped = (text or "").strip()
    if "```" not in stripped:
        return stripped
    blocks = re.findall(r"```(?:json|python|text)?\s*(.*?)```", stripped, re.S | re.I)
    return "\n".join(block.strip() for block in blocks) if blocks else stripped


def _decode_payload(text: str) -> Any:
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


def parse_assignments(text: str) -> ParseResult:
    try:
        payload = _decode_payload(text)
    except ValueError as exc:
        return ParseResult([], False, 0, 0, str(exc))

    if isinstance(payload, dict):
        raw_items = payload.get("assignments", payload.get("slots", []))
    elif isinstance(payload, list):
        raw_items = payload
    else:
        return ParseResult([], False, 0, 0, "top-level value must be an object or list")

    if not isinstance(raw_items, list):
        return ParseResult([], False, 0, 0, "assignments must be a list")

    assignments: List[Assignment] = []
    invalid = 0
    for raw in raw_items:
        if isinstance(raw, dict) and {"field", "value"}.issubset(raw):
            assignments.append(
                Assignment(raw.get("day", raw.get("days")), raw["field"], raw["value"])
            )
        elif isinstance(raw, (list, tuple)) and len(raw) == 3:
            assignments.append(Assignment(raw[0], raw[1], raw[2]))
        else:
            invalid += 1

    return ParseResult(
        assignments=assignments,
        parse_success=True,
        raw_item_count=len(raw_items),
        invalid_item_count=invalid,
    )

