"""Generic, dependency-light helper functions used throughout the project."""
from __future__ import annotations

import json
import logging
import logging.config
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def read_yaml(path: Path | str) -> dict:
    """Read a YAML file and return its contents as a dict."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    logger.debug("Loaded YAML config from %s", path)
    return content


def write_yaml(path: Path | str, data: dict) -> None:
    path = Path(path)
    create_directories([path.parent])
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: Path | str, data: dict) -> None:
    path = Path(path)
    create_directories([path.parent])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved JSON to %s", path)


def load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_directories(paths: list[Path | str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy and torch (if available) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")


def setup_logging(config_path: Path | str | None = None, default_level: int = logging.INFO) -> None:
    """Configure logging from a YAML dict-config file, falling back to a
    sensible basicConfig if the file is missing or invalid."""
    from object_detection_suite.constants.constants import LOGS_DIR

    create_directories([LOGS_DIR])

    if config_path and Path(config_path).exists():
        try:
            cfg = read_yaml(config_path)
            logging.config.dictConfig(cfg)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load logging config ({exc}); using basicConfig instead.")

    logging.basicConfig(
        level=default_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(LOGS_DIR) / "run.log"),
        ],
    )


def get_device(prefer_cuda: bool = True) -> str:
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
        if prefer_cuda and torch.backends.mps.is_available():  # Apple silicon
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Timer:
    """Small context manager to measure elapsed wall-clock time in seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
