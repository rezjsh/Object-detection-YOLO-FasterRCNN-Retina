"""Downloads the configured Roboflow dataset (default: Hard Hat Workers,
YOLOv8 format), lays it out under `data/processed/<split>/{images,labels}`,
optionally subsamples it to stay within the 2k-7k target range, and runs the
dataset validator.

Requires a Roboflow API key (free tier is enough): sign up at
https://roboflow.com, grab your key, and either:
    export ROBOFLOW_API_KEY=xxxxx
or put it in a `.env` file (see `.env.example`) and it will be picked up
automatically.

Usage:
    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --max-images 5000
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

from object_detection_suite.config.configuration import ConfigurationManager
from object_detection_suite.constants.constants import VALID_SPLITS
from object_detection_suite.data.dataset_validator import validate_dataset
from object_detection_suite.entity.config_entity import DataConfig
from object_detection_suite.utils.common import create_directories, setup_logging

logger = logging.getLogger(__name__)


def _load_api_key() -> str:
    import os

    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY"):
                _, _, value = line.partition("=")
                os.environ.setdefault("ROBOFLOW_API_KEY", value.strip())

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Create a free account at "
            "https://roboflow.com, generate an API key, then either "
            "`export ROBOFLOW_API_KEY=...` or copy `.env.example` to `.env` "
            "and fill it in."
        )
    return api_key

import shutil
from pathlib import Path

def download_from_roboflow(data_cfg: DataConfig) -> Path:
    from roboflow import Roboflow

    api_key = _load_api_key()
    
    # Wipe the raw directory completely to ensure a clean slate
    if data_cfg.raw_dir.exists():
        logger.info("Clearing existing raw directory to force a fresh download...")
        shutil.rmtree(data_cfg.raw_dir)
        
    create_directories([data_cfg.raw_dir])

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(data_cfg.roboflow.workspace).project(data_cfg.roboflow.project)
    version = project.version(data_cfg.roboflow.version)
    
    # 1. Let the SDK download to its default location by omitting `location`
    logger.info("Initiating download via Roboflow SDK...")
    dataset = version.download(data_cfg.roboflow.format)
    
    # The SDK accurately tracks where it placed the files in dataset.location
    default_location = Path(dataset.location)
    logger.info("SDK deposited files at: %s", default_location)
    
    # 2. Move the files precisely into our modular data/raw directory
    if default_location.resolve() != data_cfg.raw_dir.resolve():
        logger.info("Moving dataset from %s to %s", default_location, data_cfg.raw_dir)
        for item in default_location.iterdir():
            shutil.move(str(item), str(data_cfg.raw_dir / item.name))
        
        # Clean up the empty default folder left behind by the SDK
        shutil.rmtree(default_location)
        
    # Verify the move was successful
    contents = [p.name for p in data_cfg.raw_dir.iterdir()]
    logger.info("Final contents of raw directory: %s", contents)

    return data_cfg.raw_dir


def _find_split_root(raw_location: Path) -> Path:
    """Recursively searches for the directory containing the 'train' split."""
    # Fast path: check if it's right at the root
    if (raw_location / "train").exists():
        return raw_location
    
    # Deep search for the 'train' directory
    for path in raw_location.rglob("train"):
        if path.is_dir():
            return path.parent
            
    # Fallback if nothing is found (will trigger the missing split warnings)
    return raw_location

def organize_into_processed(raw_location: Path, processed_dir: Path) -> None:
    """Roboflow's YOLOv8 export already produces train/valid/test folders
    each containing images/ and labels/ — exactly the layout `YOLODataset`
    expects — so this is a straight copy into `data/processed/`, keeping raw
    and processed cleanly separated for reproducibility.
    
    If the export is missing the validation split, this function dynamically
    carves out a validation set from the train split to prevent pipeline crashes.
    """
    create_directories([processed_dir])
    actual_root = _find_split_root(raw_location)
    
    # 1. Copy over existing splits first
    for split in VALID_SPLITS:
        src = actual_root / split
        if not src.exists():
            logger.warning("Expected split '%s' not found in %s", split, actual_root)
            continue
        dst = processed_dir / split
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.info("Copied split '%s' -> %s", split, dst)

    # 2. Handle missing validation split dynamically
    train_processed = processed_dir / "train"
    val_processed = processed_dir / "valid"
    
    if train_processed.exists() and not val_processed.exists():
        logger.info("Validation split missing. Splitting train set to create 'valid'...")
        
        # Read the validation split ratio from configuration (defaulting to 15%)[cite: 1, 3]
        cfg_manager = ConfigurationManager()
        data_cfg = cfg_manager.get_data_config()
        val_ratio = data_cfg.val_split  # e.g., 0.15[cite: 3]
        
        train_images_dir = train_processed / "images"
        train_labels_dir = train_processed / "labels"
        
        if train_images_dir.exists() and train_labels_dir.exists():
            image_files = sorted(list(train_images_dir.iterdir()))
            
            # Shuffle deterministically using a fixed seed
            rng = random.Random(42)
            rng.shuffle(image_files)
            
            num_val = max(1, int(len(image_files) * val_ratio))
            val_images = image_files[:num_val]
            
            # Create validation directories
            val_images_dir = val_processed / "images"
            val_labels_dir = val_processed / "labels"
            create_directories([val_images_dir, val_labels_dir])
            
            # Move the subset from train to validation
            for img_path in val_images:
                # Find matching label
                lbl_path = train_labels_dir / f"{img_path.stem}.txt"
                
                # Move image
                shutil.move(str(img_path), str(val_images_dir / img_path.name))
                
                # Move label if it exists
                if lbl_path.exists():
                    shutil.move(str(lbl_path), str(val_labels_dir / lbl_path.name))
                    
            logger.info(
                "Successfully carved out validation split: moved %d images from 'train' to 'valid'", 
                num_val
            )


def subsample_dataset(processed_dir: Path, max_images: int, seed: int = 42) -> None:
    """Proportionally subsamples each split (train/valid/test) so the total
    image count across the dataset does not exceed `max_images`, keeping the
    original split ratios intact."""
    rng = random.Random(seed)
    split_files: dict[str, list[Path]] = {}
    total = 0
    for split in VALID_SPLITS:
        images_dir = processed_dir / split / "images"
        if not images_dir.exists():
            continue
        files = sorted(images_dir.iterdir())
        split_files[split] = files
        total += len(files)

    if total <= max_images:
        logger.info("Dataset already has %d images (<= max_images=%d); skipping subsample", total, max_images)
        return

    keep_ratio = max_images / total
    logger.info("Subsampling dataset from %d to ~%d images (ratio=%.3f)", total, max_images, keep_ratio)

    for split, files in split_files.items():
        n_keep = max(1, int(round(len(files) * keep_ratio)))
        rng.shuffle(files)
        to_remove = files[n_keep:]
        images_dir = processed_dir / split / "images"
        labels_dir = processed_dir / split / "labels"
        for img_path in to_remove:
            img_path.unlink(missing_ok=True)
            label_path = labels_dir / f"{img_path.stem}.txt"
            label_path.unlink(missing_ok=True)
        logger.info("Split '%s': kept %d / %d images", split, n_keep, len(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the object detection dataset")
    parser.add_argument("--max-images", type=int, default=None, help="Override the max_images cap from data.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Skip download; just re-validate/subsample existing data/processed")
    args = parser.parse_args()

    setup_logging()
    cfg_manager = ConfigurationManager()
    data_cfg = cfg_manager.get_data_config()

    raw_yaml = cfg_manager._data_raw  # noqa: SLF001 (internal, but simplest way to read max_images)
    max_images = args.max_images or raw_yaml.get("max_images")

    if not args.skip_download:
        raw_location = download_from_roboflow(data_cfg)
        organize_into_processed(raw_location, data_cfg.processed_dir)
    else:
        logger.info("Skipping download; using existing data at %s", data_cfg.processed_dir)

    if max_images:
        subsample_dataset(data_cfg.processed_dir, int(max_images))

    logger.info("Validating dataset...")
    reports = validate_dataset(data_cfg.processed_dir, num_classes=data_cfg.num_classes)
    n_errors = sum(len(r.errors) for r in reports.values())
    if n_errors:
        logger.warning("Validation finished with %d error(s) — review the log above before training.", n_errors)
    else:
        logger.info("Validation passed with no errors.")


if __name__ == "__main__":
    main()
