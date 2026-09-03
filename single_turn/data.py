"""TravelPlanner data loading and deterministic train/eval partitioning.

The collaboration task intentionally treats TravelPlanner as an environment,
not as a supervised imitation dataset.  Rows are therefore normalized only
from the query metadata and sole-planning reference information; human
``annotated_plan`` values are discarded and never reach the reward function.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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


def _load_rows(
    dataset_name: str,
    config_name: str,
    split: str,
    *,
    revision: str | None = None,
) -> List[Dict[str, Any]]:
    path = Path(dataset_name)
    if path.exists():
        split_path = path / f"{split}.jsonl" if path.is_dir() else path
        if not split_path.exists():
            raise FileNotFoundError(f"TravelPlanner split not found: {split_path}")
        return _read_jsonl(split_path)

    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        revision=revision,
    )
    return [dict(row) for row in dataset]


def normalize_travelplanner_row(
    raw: Dict[str, Any],
    index: int,
    *,
    source_split: str = "unknown",
) -> Dict[str, Any]:
    item = dict(raw)
    query = str(item.get("query") or item.get("prompt") or "").strip()
    days = int(item.get("days") or 0)
    if not query:
        raise ValueError(f"TravelPlanner row {index} has no query.")
    if days < 1:
        raise ValueError(f"TravelPlanner row {index} has invalid days={days!r}.")

    reference_records = safe_literal(item.get("reference_information"), [])
    if not isinstance(reference_records, list):
        reference_records = []
    reference_text = item.get("reference_information", "")
    if not isinstance(reference_text, str):
        reference_text = repr(reference_text)

    local_constraint = safe_literal(item.get("local_constraint"), {})
    if not isinstance(local_constraint, dict):
        local_constraint = {}

    dates = safe_literal(item.get("date"), [])
    if not isinstance(dates, list):
        dates = []

    # Do not merely leave the human plan unused: remove it from the normalized
    # row so future prompt/reward changes cannot accidentally consume it.
    item.pop("annotated_plan", None)
    item.pop("gold_plan", None)

    item.update(
        {
            "id": str(item.get("id", f"travelplanner-{source_split}-{index}")),
            "source_split": str(source_split),
            "source_index": int(index),
            "prompt": query,
            "query": query,
            "days": days,
            "dates": [str(value) for value in dates],
            "reference_information": reference_text,
            "reference_records": reference_records,
            "reference_chars": len(reference_text),
            "local_constraint": local_constraint,
            "test": "",
            "entry_point": "",
        }
    )
    return item


def _normalize_rows(
    rows: Iterable[Dict[str, Any]], *, source_split: str
) -> List[Dict[str, Any]]:
    return [
        normalize_travelplanner_row(row, idx, source_split=source_split)
        for idx, row in enumerate(rows)
    ]


def _allowed_values(value: Any, cast) -> set[Any] | None:
    if value is None:
        return None
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {cast(item) for item in values}


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    days: Any = None,
    levels: Any = None,
    visiting_city_numbers: Any = None,
    max_reference_chars: int | None = None,
    select_shortest: int | None = None,
) -> List[Dict[str, Any]]:
    """Select a reproducible curriculum subset without inspecting a gold plan."""

    allowed_days = _allowed_values(days, int)
    allowed_levels = _allowed_values(levels, lambda value: str(value).casefold())
    allowed_city_counts = _allowed_values(visiting_city_numbers, int)
    selected = [
        dict(row)
        for row in rows
        if (allowed_days is None or int(row.get("days", 0)) in allowed_days)
        and (
            allowed_levels is None
            or str(row.get("level", "")).casefold() in allowed_levels
        )
        and (
            allowed_city_counts is None
            or int(row.get("visiting_city_number", 0)) in allowed_city_counts
        )
        and (
            max_reference_chars is None
            or int(row.get("reference_chars", 0)) <= int(max_reference_chars)
        )
    ]
    if select_shortest is not None:
        count = int(select_shortest)
        if count < 1:
            raise ValueError("select_shortest must be positive when provided.")
        if count < len(selected):
            selected = sorted(
                selected,
                key=lambda row: (
                    int(row.get("reference_chars", 0)),
                    int(row.get("source_index", 0)),
                ),
            )[:count]
    return selected


def partition_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    train_samples: int,
    eval_samples: int,
    seed: int,
    stratify_by: Sequence[str] | str | None = None,
    interleave_eval: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if train_samples < 1 or eval_samples < 1:
        raise ValueError("train_samples and eval_samples must both be positive.")
    if train_samples + eval_samples > len(rows):
        raise ValueError(
            f"Requested {train_samples}+{eval_samples} rows from only {len(rows)}."
        )

    if stratify_by:
        fields = (
            (str(stratify_by),)
            if isinstance(stratify_by, str)
            else tuple(str(field) for field in stratify_by)
        )
        groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row.get(field) for field in fields)
            groups.setdefault(key, []).append(dict(row))
        ordered_keys = sorted(
            groups,
            key=lambda key: tuple(
                (
                    (0, float(value))
                    if isinstance(value, (int, float))
                    else (1, str(value))
                )
                for value in key
            ),
        )
        group_count = len(ordered_keys)
        if train_samples % group_count or eval_samples % group_count:
            raise ValueError(
                "Stratified train/eval sizes must divide evenly across "
                f"{group_count} groups."
            )
        train_per_group = train_samples // group_count
        eval_per_group = eval_samples // group_count
        rng = random.Random(seed)
        train_rows: List[Dict[str, Any]] = []
        eval_groups: List[List[Dict[str, Any]]] = []
        for key in ordered_keys:
            candidates = list(groups[key])
            if train_per_group + eval_per_group > len(candidates):
                raise ValueError(
                    f"Stratum {key!r} contains only {len(candidates)} rows, but "
                    f"needs {train_per_group + eval_per_group}."
                )
            rng.shuffle(candidates)
            eval_groups.append(candidates[:eval_per_group])
            train_rows.extend(
                candidates[eval_per_group : eval_per_group + train_per_group]
            )
        # The stock trainer uses shuffle=False. Mix the stratified train blocks
        # once deterministically so every epoch does not process four long,
        # difficulty-homogeneous runs in sequence.
        rng.shuffle(train_rows)
        if interleave_eval:
            eval_rows = [
                group[row_idx]
                for row_idx in range(eval_per_group)
                for group in eval_groups
            ]
        else:
            eval_rows = [row for group in eval_groups for row in group]
        return train_rows, eval_rows

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
    config_name: str = "validation",
    split: str = "validation",
    train_samples: int = 16,
    eval_samples: int = 4,
    seed: int = 42,
    days: Any = None,
    levels: Any = None,
    visiting_city_numbers: Any = None,
    max_reference_chars: int | None = None,
    select_shortest: int | None = None,
    stratify_by: Sequence[str] | str | None = None,
    interleave_eval: bool = False,
    revision: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load query/reference rows and make a disjoint deterministic split."""

    rows = _normalize_rows(
        _load_rows(
            dataset_name,
            config_name,
            split,
            revision=revision,
        ),
        source_split=split,
    )
    rows = filter_rows(
        rows,
        days=days,
        levels=levels,
        visiting_city_numbers=visiting_city_numbers,
        max_reference_chars=max_reference_chars,
        select_shortest=select_shortest,
    )
    if not rows:
        raise ValueError(
            "No TravelPlanner rows matched the configured curriculum filters."
        )
    return partition_rows(
        rows,
        train_samples=int(train_samples),
        eval_samples=int(eval_samples),
        seed=int(seed),
        stratify_by=stratify_by,
        interleave_eval=bool(interleave_eval),
    )
