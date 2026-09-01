"""TravelPlanner data loading and deterministic train/eval partitioning."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from single_turn.aggregation import PLAN_FIELDS


def safe_literal(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_rows(dataset_name: str, config_name: str, split: str) -> List[Dict[str, Any]]:
    path = Path(dataset_name)
    if path.exists():
        split_path = path / f"{split}.jsonl" if path.is_dir() else path
        if not split_path.exists():
            raise FileNotFoundError(f"TravelPlanner split not found: {split_path}")
        return _read_jsonl(split_path)

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name, split=split)
    return [dict(row) for row in dataset]


def _extract_gold_plan(raw: Any) -> List[Dict[str, Any]]:
    value = safe_literal(raw, [])
    if not isinstance(value, list):
        return []

    # Official train rows store [query_metadata, seven-day-plan].
    for candidate in value[1:]:
        if isinstance(candidate, list) and candidate and all(
            isinstance(day, dict) for day in candidate
        ):
            return [dict(day) for day in candidate]

    # Local fixtures may store the plan directly as a list of day dictionaries.
    if value and all(isinstance(day, dict) for day in value):
        return [dict(day) for day in value]
    return []


def _normalize_plan(plan: Sequence[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    by_day: Dict[int, Dict[str, Any]] = {}
    for fallback_day, raw_day in enumerate(plan, start=1):
        day_number = raw_day.get("day", raw_day.get("days", fallback_day))
        try:
            day_number = int(day_number)
        except (TypeError, ValueError):
            day_number = fallback_day
        if not 1 <= day_number <= days:
            continue
        normalized = {"day": day_number}
        for field in PLAN_FIELDS:
            value = raw_day.get(field, "-")
            normalized[field] = "-" if value is None else str(value).strip() or "-"
        by_day[day_number] = normalized

    return [
        by_day.get(
            day,
            {"day": day, **{field: "-" for field in PLAN_FIELDS}},
        )
        for day in range(1, days + 1)
    ]


def normalize_travelplanner_row(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    item = dict(raw)
    query = str(item.get("query") or item.get("prompt") or "").strip()
    days = int(item.get("days") or 0)
    if not query:
        raise ValueError(f"TravelPlanner row {index} has no query.")
    if days < 1:
        raise ValueError(f"TravelPlanner row {index} has invalid days={days!r}.")

    extracted_gold_plan = _extract_gold_plan(item.get("annotated_plan"))
    if not extracted_gold_plan:
        raise ValueError(
            "The phase-one MAGRPO reward requires annotated TravelPlanner plans; "
            f"row {index} has none. Use the official train configuration."
        )
    gold_plan = _normalize_plan(extracted_gold_plan, days)

    reference_records = safe_literal(item.get("reference_information"), [])
    if not isinstance(reference_records, list):
        reference_records = []
    reference_text = item.get("reference_information", "")
    if not isinstance(reference_text, str):
        reference_text = repr(reference_text)

    local_constraint = safe_literal(item.get("local_constraint"), {})
    if not isinstance(local_constraint, dict):
        local_constraint = {}

    item.update(
        {
            "id": str(item.get("id", f"travelplanner-train-{index}")),
            "prompt": query,
            "query": query,
            "days": days,
            "gold_plan": gold_plan,
            "reference_information": reference_text,
            "reference_records": reference_records,
            "local_constraint": local_constraint,
            "test": "",
            "entry_point": "",
        }
    )
    return item


def _normalize_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_travelplanner_row(row, idx) for idx, row in enumerate(rows)]


def partition_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    train_samples: int,
    eval_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if train_samples < 1 or eval_samples < 1:
        raise ValueError("train_samples and eval_samples must both be positive.")
    if train_samples + eval_samples > len(rows):
        raise ValueError(
            f"Requested {train_samples}+{eval_samples} rows from only {len(rows)}."
        )
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    eval_indices = indices[:eval_samples]
    train_indices = indices[eval_samples : eval_samples + train_samples]
    return (
        [dict(rows[idx]) for idx in train_indices],
        [dict(rows[idx]) for idx in eval_indices],
    )


def load_single_turn_datasets(
    dataset_name: str,
    *,
    config_name: str = "train",
    split: str = "train",
    train_samples: int = 40,
    eval_samples: int = 5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load annotated plans and make a disjoint deterministic 40/5 split."""

    rows = _normalize_rows(_load_rows(dataset_name, config_name, split))
    return partition_rows(
        rows,
        train_samples=int(train_samples),
        eval_samples=int(eval_samples),
        seed=int(seed),
    )
