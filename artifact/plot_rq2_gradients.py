#!/usr/bin/env python3
"""Reproduce Figure 2 from the corrected per-step gradient history."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SERIES = (
    ("joint_embd_corrected_cosine_similarity", "joint_embd_CLAP", "#1f77b4", "-"),
    ("joint_sim_corrected_cosine_similarity", "joint_sim_CLAP", "#d62728", "--"),
    ("joint_rank_corrected_cosine_similarity", "joint_rank_CLAP", "#2ca02c", "-."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("artifact/rq2_gradient_history.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ema-weight",
        type=float,
        default=0.98,
        help="EMA smoothing weight used for the prominent trend lines",
    )
    parser.add_argument(
        "--preview-png",
        type=Path,
        help="Optional raster preview for visual inspection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.ema_weight < 1:
        raise ValueError("--ema-weight must be in [0, 1)")

    history = pd.read_csv(args.history)
    required = {"global_step", *(column for column, _, _, _ in SERIES)}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Gradient history is missing columns: {sorted(missing)}")

    epoch = history["global_step"] / history["global_step"].iloc[-1]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(4.5, 3.2))
    for column, label, color, linestyle in SERIES:
        values = history[column]
        trend = values.ewm(alpha=1 - args.ema_weight, adjust=True).mean()
        axis.plot(epoch, values, color=color, alpha=0.12, linewidth=0.45)
        axis.plot(
            epoch,
            trend,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=label,
        )

    axis.axhline(0, color="black", linewidth=0.7, alpha=0.55)
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.65, 0.65)
    axis.set_xlabel("Epoch", labelpad=4)
    axis.set_ylabel("Cosine Similarity", labelpad=4)
    axis.set_title("Gradient Interaction During Joint Training", pad=7)
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.45, color="gray")
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout(pad=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="pdf", bbox_inches="tight")
    if args.preview_png:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.preview_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
