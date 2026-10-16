"""Optional checkpoint arguments and W&B identity for training entrypoints."""

from collections.abc import Mapping
import os


def checkpoint_train_kwargs(config):
    """Return only explicitly configured checkpoint options, without reading files."""
    section = config.get("checkpoint")
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError("checkpoint must be a mapping of train() options.")
    paths = {"resume_from_checkpoint", "checkpoint_dir"}
    counts = {"checkpoint_interval", "checkpoint_total_limit"}
    unknown = set(section) - paths - counts
    if unknown:
        raise ValueError(f"Unknown checkpoint options: {sorted(map(str, unknown))}")
    result = {}
    for key, value in section.items():
        if value is None:
            if key == "checkpoint_interval":
                raise ValueError(
                    "checkpoint.checkpoint_interval must be a positive integer."
                )
            continue
        if key in paths:
            if not isinstance(value, (str, os.PathLike)):
                raise ValueError(f"checkpoint.{key} must be a nonempty path.")
            value = os.fspath(value)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"checkpoint.{key} must be a nonempty path.")
        elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"checkpoint.{key} must be a positive integer.")
        result[key] = value
    return result


def prepare_wandb_config(config, wandb_config, checkpoint_options):
    """Apply explicit logging choices or a saved identity before trainer creation."""
    section = config.get("wandb", {})
    if wandb_config is None or section is None or section is False:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("wandb must be a mapping, null, or false.")
    enabled = section.get("enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
    if (enabled is not None and not enabled) or section.get("mode") == "disabled":
        return None
    explicit = {key: section[key] for key in ("id", "resume") if key in section}
    result = {**wandb_config, **explicit} if explicit else wandb_config
    if "id" in result or "resume" in result:
        return result
    checkpoint = checkpoint_options.get("resume_from_checkpoint")
    if checkpoint is None:
        return result
    from comlrl.utils import read_checkpoint_metadata

    identity = read_checkpoint_metadata(checkpoint)["wandb"]
    if identity:
        return {**result, **identity, "resume": "allow"}
    return result
