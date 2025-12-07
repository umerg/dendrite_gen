"""Hydra config loading helpers shared by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_hydra_config(config_path: str | Path, overrides: Sequence[str] | None = None) -> DictConfig:
    """Load a Hydra config from disk, applying optional overrides.

    Args:
        config_path: Path to a YAML config (e.g., ``config/small_trees_run.yaml``).
        overrides: Optional iterable of Hydra override strings.

    Returns:
        A resolved ``DictConfig`` ready for downstream use.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config_dir = path.parent
    config_name = path.stem
    overrides = list(overrides or [])
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
