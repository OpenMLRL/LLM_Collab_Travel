"""Configuration helpers for single-turn TravelPlanner experiments."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    type: str = "qwen"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    special_tokens: Dict[str, str] = field(default_factory=dict)
    torch_dtype: Optional[str] = None
    attn_implementation: Optional[str] = "sdpa"

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelConfig":
        missing = [key for key in ("name", "temperature", "top_p") if key not in raw]
        if missing:
            raise ValueError(
                "agent_model is missing required fields: " + ", ".join(missing)
            )

        def optional_float(value: Any) -> Optional[float]:
            return None if value is None else float(value)

        def optional_int(value: Any) -> Optional[int]:
            if value is None or str(value).strip().lower() in {"", "none", "null"}:
                return None
            return int(value)

        return cls(
            name=str(raw["name"]),
            type=str(raw.get("type", "qwen")),
            temperature=optional_float(raw.get("temperature")),
            top_p=optional_float(raw.get("top_p")),
            top_k=optional_int(raw.get("top_k")),
            special_tokens=dict(raw.get("special_tokens", {})),
            torch_dtype=raw.get("torch_dtype") or raw.get("dtype"),
            attn_implementation=raw.get("attn_implementation", "sdpa"),
        )


class Config:
    def __init__(self, config_path: str):
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with self.path.open("r", encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle) or {}

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        value = self.data.get(section, {})
        return dict(value) if isinstance(value, dict) else {}

    def get_agent_model_config(self) -> ModelConfig:
        section = self.get_section("agent_model")
        if not section:
            raise ValueError("No 'agent_model' section found in configuration")
        return ModelConfig.from_dict(section)

    def update(self, updates: Dict[str, Any]) -> None:
        self._deep_merge(self.data, updates)

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                cls._deep_merge(base[key], value)
            else:
                base[key] = value

    def save(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.data, handle, sort_keys=False)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--override", nargs="*", help="Override YAML values with key=value entries."
    )
    return parser


def _parse_override_value(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_overrides(overrides: Optional[list[str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, raw_value = override.split("=", 1)
        current = result
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = _parse_override_value(raw_value)
    return result
