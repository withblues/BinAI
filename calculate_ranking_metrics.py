"""Calculate exact full-corpus retrieval metrics from stored embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

from datasets import load_from_disk
import numpy as np
import torch

from src.utils.ranking_metrics import evaluate_embeddings


IDENTITY_COLUMNS = ("binary_name", "function_name")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _load_eligible_ids(path: str | None) -> set[Any] | None:
    if path is None:
        return None
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        if "unique_id" not in dataset.column_names:
            raise ValueError(
                f"eligible-ID dataset {path!r} does not contain a unique_id column"
            )
        values = np.asarray(dataset["unique_id"])
    else:
        values = np.load(path, allow_pickle=True)
    if values.ndim != 1:
        raise ValueError(
            "--eligible_ids_path must be a Hugging Face dataset or contain a "
            "one-dimensional NumPy array"
        )
    result = {_python_scalar(value) for value in values}
    if len(result) != len(values):
        raise ValueError("--eligible_ids_path contains duplicate IDs")
    return result


def _load_embeddings(
    embeddings_dataset_path: str,
    test_ids: set[Any],
    eligible_ids: set[Any] | None,
    embedding_column: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_from_disk(embeddings_dataset_path)
    if "unique_id" not in dataset.column_names:
        raise ValueError("embedding dataset is missing the unique_id column")
    if embedding_column is None:
        if "embedding" in dataset.column_names:
            embedding_column = "embedding"
        else:
            candidates = [
                column for column in dataset.column_names if column.endswith("_embedding")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "embedding dataset has no 'embedding' column and automatic detection "
                    f"found {candidates}; pass --embedding_column explicitly"
                )
            embedding_column = candidates[0]
            print(f"Using detected embedding column {embedding_column!r}.")
    elif embedding_column not in dataset.column_names:
        raise ValueError(
            f"requested embedding column {embedding_column!r} is absent; available "
            f"columns: {dataset.column_names}"
        )
    dataset.set_format("numpy", columns=["unique_id", embedding_column])
    values = dataset[:]
    ids = np.asarray(values["unique_id"])
    embeddings = np.asarray(values[embedding_column])
    normalized_ids = [_python_scalar(value) for value in ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("embedding dataset contains duplicate unique_id values")

    available_ids = set(normalized_ids)
    if eligible_ids is not None:
        missing_eligible = eligible_ids - available_ids
        if missing_eligible:
            preview = list(missing_eligible)[:10]
            raise ValueError(
                f"embedding dataset is missing {len(missing_eligible)} IDs required by "
                f"--eligible_ids_path; first IDs: {preview}"
            )

    keep = np.asarray(
        [
            uid in test_ids and (eligible_ids is None or uid in eligible_ids)
            for uid in normalized_ids
        ],
        dtype=bool,
    )
    if not keep.any():
        raise ValueError("no embedding IDs remain after test/eligible-ID filtering")
    dropped = len(keep) - int(keep.sum())
    if dropped:
        print(f"Filtered out {dropped:,} embeddings outside the evaluation population.")
    return ids[keep], np.ascontiguousarray(embeddings[keep], dtype=np.float32)


def _load_group_keys(
    data_dir: str,
    ids: np.ndarray,
    *,
    metadata_num_proc: int,
) -> list[tuple[str, str, str]]:
    metadata_path = os.path.join(data_dir, "assembly_x64_1024_clap")
    metadata = load_from_disk(metadata_path)
    missing = {"unique_id", *IDENTITY_COLUMNS} - set(metadata.column_names)
    if missing:
        raise ValueError(f"metadata dataset is missing columns: {sorted(missing)}")

    wanted = {_python_scalar(value) for value in ids}
    metadata = metadata.filter(
        lambda batch: [_python_scalar(uid) in wanted for uid in batch["unique_id"]],
        batched=True,
        num_proc=None if metadata_num_proc == 1 else metadata_num_proc,
        desc="Selecting evaluation metadata",
    )
    metadata.set_format("numpy", columns=["unique_id", *IDENTITY_COLUMNS])
    values = metadata[:]

    by_id: dict[Any, tuple[str, str, str]] = {}
    for row in range(len(metadata)):
        uid = _python_scalar(values["unique_id"][row])
        if uid in by_id:
            raise ValueError(f"metadata contains duplicate unique_id {uid!r}")
        by_id[uid] = tuple(str(values[column][row]) for column in IDENTITY_COLUMNS)

    missing_ids = [
        _python_scalar(uid) for uid in ids if _python_scalar(uid) not in by_id
    ]
    if missing_ids:
        raise ValueError(
            f"metadata is missing {len(missing_ids)} embedding IDs; first IDs: {missing_ids[:10]}"
        )
    return [by_id[_python_scalar(uid)] for uid in ids]


def _select_benchmark_queries(
    group_keys: list[tuple[str, str, str]], count: int, seed: int
) -> np.ndarray:
    frequencies = Counter(group_keys)
    eligible = np.asarray(
        [index for index, key in enumerate(group_keys) if frequencies[key] > 1],
        dtype=np.int64,
    )
    if count > len(eligible):
        raise ValueError(
            f"requested {count} benchmark queries, but only {len(eligible)} are eligible"
        )
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=count, replace=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact MRR, Recall@k, and nDCG@k over the full non-self "
            "candidate corpus without materializing or sorting the full score matrix."
        )
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="project")
    parser.add_argument("--method", required=True, help="Model/run label used in the report")
    parser.add_argument("--model_name", default="clap", help="Teacher/report grouping label")
    parser.add_argument("--embeddings_dataset_path", required=True)
    parser.add_argument(
        "--embedding_column",
        help=(
            "Embedding column name. Defaults to 'embedding', or the sole column "
            "ending in '_embedding'."
        ),
    )
    parser.add_argument(
        "--batch_size",
        "--query_batch_size",
        dest="query_batch_size",
        default=128,
        type=int,
        help="Queries processed together; lower this if memory is insufficient",
    )
    parser.add_argument(
        "--candidate_block_size",
        default=32768,
        type=int,
        help="Corpus embeddings scored per block",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="PyTorch CPU worker threads; defaults to the environment configuration",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 512, 1024],
        help="Recall and nDCG cutoffs",
    )
    parser.add_argument(
        "--eligible_ids_path",
        help=(
            "Optional .npy file or Hugging Face embedding dataset defining an exact "
            "common evaluation population across models"
        ),
    )
    parser.add_argument(
        "--benchmark_queries",
        type=int,
        help="Evaluate a seeded subset and project full-population runtime",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata_num_proc", type=int, default=16)
    parser.add_argument("--report_path", help="Override the default JSON report path")
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable the query-batch progress bar",
    )
    parser.add_argument(
        "--skip_teacher",
        action="store_true",
        help="Deprecated compatibility flag; teacher scores are no longer loaded",
    )
    parser.add_argument("--limit", type=int, help="Deprecated alias for --benchmark_queries")
    args = parser.parse_args()
    if args.limit is not None:
        if args.benchmark_queries is not None:
            parser.error("use only one of --limit and --benchmark_queries")
        args.benchmark_queries = args.limit
    return args


def main() -> None:
    args = parse_args()
    if args.metadata_num_proc <= 0:
        raise ValueError("--metadata_num_proc must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    selected_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if selected_device == "auto":
        selected_device = "cpu"
    print(f"Evaluating {args.method!r} on {selected_device} with split {args.split!r}.")

    split_path = os.path.join(args.data_dir, f"cross_{args.split}_split.json")
    with open(split_path, encoding="utf-8") as stream:
        split_definition = json.load(stream)
    if "test" not in split_definition:
        raise ValueError(f"{split_path} has no 'test' split")
    test_ids = {_python_scalar(value) for value in split_definition["test"]}
    eligible_ids = _load_eligible_ids(args.eligible_ids_path)
    ids, embeddings = _load_embeddings(
        args.embeddings_dataset_path,
        test_ids,
        eligible_ids,
        args.embedding_column,
    )
    group_keys = _load_group_keys(
        args.data_dir, ids, metadata_num_proc=args.metadata_num_proc
    )
    print(f"Loaded {len(ids):,} embeddings with dimension {embeddings.shape[1]:,}.")

    query_indices = None
    if args.benchmark_queries is not None:
        if args.benchmark_queries <= 0:
            raise ValueError("--benchmark_queries must be positive")
        query_indices = _select_benchmark_queries(group_keys, args.benchmark_queries, args.seed)
        print(f"Benchmarking {len(query_indices):,} seeded eligible queries.")

    evaluation = evaluate_embeddings(
        embeddings,
        group_keys,
        k_values=args.k_values,
        query_indices=query_indices,
        query_batch_size=args.query_batch_size,
        candidate_block_size=args.candidate_block_size,
        device=args.device,
        num_threads=args.num_threads,
        progress=not args.no_progress,
    )
    projected_seconds = (
        evaluation.elapsed_seconds
        * evaluation.eligible_queries
        / evaluation.evaluated_queries
    )

    report = {
        "model_name": args.method,
        "protocol": {
            "similarity": "cosine (explicit L2 normalization)",
            "self_match": "excluded before ranking and relevance counting",
            "ground_truth_key": list(IDENTITY_COLUMNS),
            "query_policy": "queries with at least one non-self positive",
            "mrr": "mean reciprocal rank of first relevant candidate",
            "relevance": "binary",
            "tie_policy": evaluation.tie_policy,
            "k_values": sorted(set(args.k_values)),
            "seed": args.seed,
            "eligible_ids_path": args.eligible_ids_path,
            "embedding_column": args.embedding_column or "auto",
        },
        "population": {
            key: value
            for key, value in asdict(evaluation).items()
            if key not in {"metrics", "tie_policy"}
        },
        "metrics": evaluation.metrics,
        "projected_full_runtime_seconds": projected_seconds,
    }
    report.update(evaluation.metrics)

    print("\n--- Corrected Retrieval Metrics ---")
    for name, value in evaluation.metrics.items():
        print(f"{name}: {value:.6f}")
    print(
        f"Evaluated {evaluation.evaluated_queries:,}/{evaluation.eligible_queries:,} "
        f"eligible queries in {evaluation.elapsed_seconds / 60:.2f} minutes."
    )
    if evaluation.evaluated_queries < evaluation.eligible_queries:
        print(f"Projected full runtime: {projected_seconds / 3600:.2f} hours.")

    report_path = args.report_path
    if report_path is None:
        report_path = os.path.join(
            args.output_dir,
            "inference",
            "metrics",
            args.split,
            args.model_name,
            f"{args.method}_corrected_metrics_report.json",
        )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
