# Object Detection Suite

A modular, multi-model object detection pipeline: one shared data/training/
evaluation pipeline, three swappable detector families, fair benchmarking,
and image/video/Streamlit inference — built with **PyTorch**, **OpenCV**,
**Albumentations**, and managed with **uv**.

## Dataset

**[Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers)**
(Roboflow Universe, `joseph-nelson/hard-hat-workers`, version 2, YOLOv8 format).

Why this dataset:
- **~7,035 images** — sits at the top of the 2k-7k "medium" target range.
  `scripts/download_dataset.py` subsamples down to `max_images` (default
  7,000) so the final dataset always lands inside the target.
- **3 classes only** (`helmet`, `head`, `person`) — easy to compare model
  families fairly without long-tail class imbalance.
- Real workplace-safety imagery with dense, varied box sizes (small
  helmets/heads, larger person boxes) — a good stress test for both
  single-stage and two-stage detectors.
- Native YOLOv8 `.txt` export — zero reformatting needed.
- License: CC BY 4.0.

## Architecture

```
configs/            YAML configs (project, data, train, eval, inference, logging)
data/                raw/ processed/ annotations/  (created by download script)
src/object_detection_suite/
  constants/         path & default-value constants
  entity/            typed dataclass config objects
  config/            ConfigurationManager: YAML -> typed configs
  utils/             yaml/json IO, seeding, logging, device helpers
  data/              dataset_registry (strategy pattern), dataset_loader, validator, augmentations
  models/            base interface, model_factory (factory pattern), yolo/faster_rcnn/retinanet
  train/             trainer (model-agnostic loop), losses (YOLO loss), checkpointing
  eval/              metrics (mAP/precision/recall/IoU from scratch), evaluator, benchmark
  infer/             predictor (image/folder), video_inference
  visualize/         annotate (OpenCV drawing), plots (matplotlib/seaborn)
app/                 Streamlit demo
scripts/             download_dataset, train_all, evaluate_all, export_model
tests/               unit tests for dataset, models, metrics
main.py              single CLI entrypoint
```

**Design patterns used:**
- **Factory pattern** — `ModelFactory.create(name, ...)` builds any of the
  three detectors from a string name; adding a fourth model means adding one
  file + one registry entry, nothing else changes.
- **Strategy pattern** — `data/dataset_registry.py` lets new dataset formats
  be registered independently of the training code.
- **Typed config objects** — every module receives a frozen dataclass, never
  raw YAML/dict, via the central `ConfigurationManager`.

### Models

| Model | Type | Notes |
|---|---|---|
| `yolo_style` | single-stage, from-scratch | ResNet-18 backbone (stride 32) + custom grid/anchor head, YOLOv2-style loss (objectness + box regression + classification), NMS decode. No `ultralytics` dependency — trains through the same shared `Trainer`. |
| `faster_rcnn` | two-stage | torchvision `fasterrcnn_resnet50_fpn`, box predictor head swapped to the dataset's class count. |
| `retinanet` | single-stage, focal loss | torchvision `retinanet_resnet50_fpn_v2`, classification head swapped to the dataset's class count. |

All three share the exact same `DataLoader`, image size (640×640), and
`Trainer`/`Evaluator`, and expose one interface: `forward(images, targets)`
returns a `{"loss": ...}` dict in train mode and a list of
`{"boxes","scores","labels"}` dicts in eval mode.

## Setup (uv)

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
uv sync                      # creates .venv and installs all dependencies from pyproject.toml

cp .env.example .env
# edit .env and set ROBOFLOW_API_KEY (free key at https://roboflow.com)
```

> **CPU-only machine?** PyPI's default `torch`/`torchvision` wheels bundle the
> full NVIDIA CUDA runtime (several GB) even if you have no GPU. If `uv sync`
> is slow or you're low on disk, install the CPU-only build first, then sync
> the rest:
> ```bash
> uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> uv sync --no-install-package torch --no-install-package torchvision
> ```
> On Colab or any CUDA machine, the default `uv sync` is exactly what you want.

## Usage

```bash
# 1. Download + validate the dataset (subsampled to <= 7,000 images)
uv run python main.py download

# 2. Train (all 3 models, or just one)
uv run python main.py train
uv run python main.py train --model yolo_style

# 3. Evaluate / benchmark all trained models on the test split
uv run python main.py evaluate
#    -> artifacts/benchmarks/benchmark_results.{csv,md} + benchmark_comparison.png

# 4. Inference
uv run python main.py infer-image  --input path/to/img.jpg --model yolo_style
uv run python main.py infer-folder --input data/samples --output artifacts/predictions --model faster_rcnn
uv run python main.py infer-video  --input path/to/video.mp4 --output out.mp4 --model retinanet

# 5. Streamlit demo
uv run python main.py app
# or directly: uv run streamlit run app/app.py

# Run tests
uv run pytest -v
```

Equivalent `make` targets: `make setup`, `make download`, `make train`,
`make train-yolo_style`, `make evaluate`, `make app`, `make test`.

## Run order (Colab or local)

1. `uv sync`
2. Set `ROBOFLOW_API_KEY` (env var or `.env`)
3. `uv run python main.py download`
4. `uv run python main.py train --model yolo_style` (fastest; try this first on a free Colab GPU)
5. `uv run python main.py train --model faster_rcnn`
6. `uv run python main.py train --model retinanet`
7. `uv run python main.py evaluate`
8. `uv run python main.py infer-image --input <your image>`

## Configuration

All hyperparameters live in `configs/*.yaml` — no code changes needed to:
- swap which models train (`train.yaml: models_to_train`)
- change image size / batch size / dataset split ratios (`data.yaml`)
- change score/NMS thresholds for eval vs. inference (`eval.yaml`, `inference.yaml`)
- point at a different Roboflow dataset (`data.yaml: roboflow.{workspace,project,version}`)

## Notes on the YOLO-style model

This project implements a compact, from-scratch YOLO-style detector
(grid + anchors + objectness + NMS, YOLOv2-style loss) rather than wrapping
the `ultralytics` package. That keeps every model trainable through the
same `Trainer`/`Evaluator`/checkpoint format — swapping models is a config
change, not a different training system. It is not a reproduction of any
specific modern YOLO release's architecture or benchmark numbers.
