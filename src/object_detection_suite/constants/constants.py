"""Central constants for the object detection suite.

Keeping these in one place avoids magic strings/numbers scattered across the
codebase and gives a single point of truth for default paths and values.
"""
from pathlib import Path

# --- Project root -------------------------------------------------------------
# constants.py lives at <root>/src/object_detection_suite/constants/constants.py
ROOT_DIR = Path(__file__).resolve().parents[3]

# --- Config files ---------------------------------------------------------------
CONFIG_DIR = ROOT_DIR / "configs"
PROJECT_CONFIG_FILE = CONFIG_DIR / "project.yaml"
DATA_CONFIG_FILE = CONFIG_DIR / "data.yaml"
TRAIN_CONFIG_FILE = CONFIG_DIR / "train.yaml"
EVAL_CONFIG_FILE = CONFIG_DIR / "eval.yaml"
INFERENCE_CONFIG_FILE = CONFIG_DIR / "inference.yaml"
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging.yaml"

# --- Data directories -------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# --- Output directories -------------------------------------------------------------
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# --- Dataset split file names (YOLO convention) -------------------------------------
IMAGES_DIR_NAME = "images"
LABELS_DIR_NAME = "labels"
TRAIN_SPLIT = "train"
VAL_SPLIT = "valid"
TEST_SPLIT = "test"
VALID_SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --- Model registry keys --------------------------------------------------------------
MODEL_YOLO = "yolo_style"
MODEL_FASTER_RCNN = "faster_rcnn"
MODEL_RETINANET = "retinanet"
SUPPORTED_MODELS = (MODEL_YOLO, MODEL_FASTER_RCNN, MODEL_RETINANET)

# --- Misc defaults ----------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_IMG_SIZE = 640
DEFAULT_NUM_WORKERS = 2
