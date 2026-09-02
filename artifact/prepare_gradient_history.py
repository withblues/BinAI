#!/usr/bin/env python3
"""Merge W&B gradient exports and recover normalized cosine similarities.

The historical ``cos_sim_distill`` series was logged as the unnormalized
gradient dot product.  The true cosine is recoverable from that series and the
two gradient-norm series exported on the same global-step grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_SUFFIXES = {
    "raw_dot_product": "train/train/cos_sim_distill",
    "infonce_grad_norm": "train/train/grad_norm_main",
    "distill_grad_norm": "train/train/grad_norm_distill",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dot-export", type=Path, required=True)
    parser.add_argument("--main-norm-export", type=Path, required=True)
    parser.add_argument("--distill-norm-export", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def objective_name(column: str) -> str:
    if "pairwiserank" in column:
        return "joint_rank_CLAP"
    if "with_proj" in column:
        return "joint_embd_CLAP"
    return "joint_sim_CLAP"


def metric_columns(frame: pd.DataFrame, suffix: str) -> dict[str, str]:
    columns = {
        objective_name(column): column
        for column in frame.columns
        if column.endswith(suffix)
        and not column.endswith("__MIN")
        and not column.endswith("__MAX")
    }
    expected = {"joint_sim_CLAP", "joint_rank_CLAP", "joint_embd_CLAP"}
    if set(columns) != expected:
        raise ValueError(
            f"Expected columns for {sorted(expected)}, found {sorted(columns)}"
        )
    return columns


def main() -> None:
    args = parse_args()
    frames = {
        "raw_dot_product": pd.read_csv(args.dot_export),
        "infonce_grad_norm": pd.read_csv(args.main_norm_export),
        "distill_grad_norm": pd.read_csv(args.distill_norm_export),
    }
    columns = {
        metric: metric_columns(frame, METRIC_SUFFIXES[metric])
        for metric, frame in frames.items()
    }

    steps = frames["raw_dot_product"]["train/global_step"]
    for metric, frame in frames.items():
        if not steps.equals(frame["train/global_step"]):
            raise ValueError(f"Global-step grid differs in {metric} export")

    merged = pd.DataFrame({"global_step": steps})
    summary: dict[str, object] = {
        "rows": int(len(steps)),
        "first_global_step": int(steps.iloc[0]),
        "last_global_step": int(steps.iloc[-1]),
        "note": (
            "The historical W&B field cos_sim_distill stored the raw gradient "
            "dot product. corrected_cosine_similarity divides it by both "
            "exported gradient norms."
        ),
        "objectives": {},
    }

    for objective in ("joint_sim_CLAP", "joint_rank_CLAP", "joint_embd_CLAP"):
        raw_dot = frames["raw_dot_product"][columns["raw_dot_product"][objective]]
        main_norm = frames["infonce_grad_norm"][columns["infonce_grad_norm"][objective]]
        distill_norm = frames["distill_grad_norm"][columns["distill_grad_norm"][objective]]
        if (main_norm <= 0).any() or (distill_norm <= 0).any():
            raise ValueError(f"Non-positive gradient norm found for {objective}")

        cosine = raw_dot / (main_norm * distill_norm)
        norm_ratio = distill_norm / main_norm
        if ((cosine < -1.00001) | (cosine > 1.00001)).any():
            raise ValueError(f"Recovered cosine outside [-1, 1] for {objective}")

        prefix = objective.removesuffix("_CLAP")
        merged[f"{prefix}_raw_dot_product"] = raw_dot
        merged[f"{prefix}_infonce_grad_norm"] = main_norm
        merged[f"{prefix}_distill_grad_norm"] = distill_norm
        merged[f"{prefix}_corrected_cosine_similarity"] = cosine
        merged[f"{prefix}_norm_ratio"] = norm_ratio

        summary["objectives"][objective] = {
            "mean_corrected_cosine_similarity": float(cosine.mean()),
            "mean_norm_ratio": float(norm_ratio.mean()),
            "mean_raw_dot_product": float(raw_dot.mean()),
            "minimum_corrected_cosine_similarity": float(cosine.min()),
            "maximum_corrected_cosine_similarity": float(cosine.max()),
        }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
