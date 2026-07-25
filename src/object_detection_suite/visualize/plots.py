"""Matplotlib/Seaborn visualizations for training curves, dataset class
distribution, and cross-model benchmark comparisons. Every function saves a
PNG to `save_path` and also returns the created Figure for interactive use
(e.g. in a notebook)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for scripts/servers
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_training_curves(history: list[dict], save_path: Path | str, model_name: str = "") -> plt.Figure:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_map = [h.get("val_mAP_50", 0.0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, train_loss, marker="o", color="tab:blue")
    axes[0].set_title(f"{model_name} — Training Loss".strip(" —"))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, val_map, marker="o", color="tab:green")
    axes[1].set_title(f"{model_name} — Validation mAP@0.5".strip(" —"))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    return fig


def plot_class_distribution(class_counts: dict[int, int], class_names: list[str], save_path: Path | str) -> plt.Figure:
    names = [class_names[cid] if cid < len(class_names) else str(cid) for cid in class_counts]
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=names, y=counts, ax=ax, hue=names, legend=False, palette="viridis")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of boxes")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    return fig


def plot_benchmark_comparison(rows: list[dict], save_path: Path | str) -> plt.Figure:
    """`rows` is a list of dicts like
    {"model": "yolo_style", "mAP_50": 0.71, "mAP_50_95": 0.42, "fps": 55.2}."""
    models = [r["model"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    sns.barplot(x=models, y=[r["mAP_50"] for r in rows], ax=axes[0], hue=models, legend=False, palette="mako")
    axes[0].set_title("mAP@0.5")
    axes[0].set_ylim(0, 1)

    sns.barplot(x=models, y=[r["mAP_50_95"] for r in rows], ax=axes[1], hue=models, legend=False, palette="mako")
    axes[1].set_title("mAP@0.5:0.95")
    axes[1].set_ylim(0, 1)

    sns.barplot(x=models, y=[r["fps"] for r in rows], ax=axes[2], hue=models, legend=False, palette="mako")
    axes[2].set_title("Inference FPS")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    return fig
