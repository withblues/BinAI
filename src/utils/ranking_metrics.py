"""Exact, memory-bounded retrieval metrics for binary-function embeddings.

The evaluator never materializes the full query-by-corpus similarity matrix.
It scans the corpus in blocks, retaining only the largest requested cutoff and
counting the candidates that outrank each query's best positive for exact MRR.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import time
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass(frozen=True)
class RankingEvaluation:
    """Aggregate metrics and population statistics for one evaluation run."""

    metrics: dict[str, float]
    total_items: int
    eligible_queries: int
    evaluated_queries: int
    excluded_singletons: int
    mean_positives_per_query: float
    candidate_pool_size: int
    elapsed_seconds: float
    tie_policy: str = "optimistic"


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"embeddings must be a 2-D array, got shape {array.shape}")
    if len(array) < 2:
        raise ValueError("at least two embeddings are required")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain NaN or infinite values")
    norms = np.linalg.norm(array, axis=1)
    zero_rows = np.flatnonzero(norms == 0)
    if len(zero_rows):
        preview = zero_rows[:10].tolist()
        raise ValueError(f"zero-norm embeddings at rows {preview}")
    return np.ascontiguousarray(array)


def _encode_group_keys(group_keys: Sequence[object]) -> tuple[np.ndarray, dict[int, list[int]]]:
    if len(group_keys) == 0:
        raise ValueError("group_keys must not be empty")

    key_to_group: dict[object, int] = {}
    members: dict[int, list[int]] = defaultdict(list)
    encoded = np.empty(len(group_keys), dtype=np.int64)
    for index, key in enumerate(group_keys):
        try:
            group = key_to_group.setdefault(key, len(key_to_group))
        except TypeError as error:
            raise TypeError("each ground-truth group key must be hashable") from error
        encoded[index] = group
        members[group].append(index)
    return encoded, dict(members)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _positive_score_thresholds(
    embeddings: torch.Tensor,
    query_indices: torch.Tensor,
    positive_lists: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Return the highest non-self positive score for every query."""
    max_positives = max(len(indices) for indices in positive_lists)
    padded = torch.zeros(
        (len(positive_lists), max_positives), dtype=torch.long, device=embeddings.device
    )
    valid = torch.zeros_like(padded, dtype=torch.bool)
    for row, indices in enumerate(positive_lists):
        count = len(indices)
        padded[row, :count] = torch.as_tensor(indices, device=embeddings.device)
        valid[row, :count] = True

    positive_embeddings = embeddings[padded]
    queries = embeddings[query_indices].unsqueeze(1)
    scores = torch.sum(queries * positive_embeddings, dim=-1)
    scores.masked_fill_(~valid, -torch.inf)
    return scores.max(dim=1).values


def _dcg_from_binary_relevance(relevance: torch.Tensor, k: int) -> torch.Tensor:
    cutoff = min(k, relevance.shape[1])
    discounts = 1.0 / torch.log2(
        torch.arange(2, cutoff + 2, device=relevance.device, dtype=torch.float32)
    )
    return torch.sum(relevance[:, :cutoff].float() * discounts, dim=1)


def evaluate_embeddings(
    embeddings: np.ndarray,
    group_keys: Sequence[object],
    *,
    k_values: Iterable[int] = (1, 512, 1024),
    query_indices: Sequence[int] | None = None,
    query_batch_size: int = 128,
    candidate_block_size: int = 32768,
    device: str = "auto",
    num_threads: int | None = None,
    progress: bool = False,
) -> RankingEvaluation:
    """Compute exact MRR, Recall@k, and binary nDCG@k.

    ``group_keys`` define relevance. Every other item with the same key is
    relevant; the query itself is always excluded. MRR uses the reciprocal rank
    of the first relevant candidate. Score ties use the optimistic rank
    ``1 + count(score > best_positive_score)`` and are documented in the result.
    """
    array = _validate_embeddings(embeddings)
    if len(group_keys) != len(array):
        raise ValueError("group_keys and embeddings must have equal length")
    if query_batch_size <= 0 or candidate_block_size <= 0:
        raise ValueError("batch sizes must be positive")

    cutoffs = sorted(set(int(k) for k in k_values))
    if not cutoffs or cutoffs[0] <= 0:
        raise ValueError("k_values must contain positive integers")

    group_ids_np, group_members = _encode_group_keys(group_keys)
    group_counts = Counter(group_ids_np.tolist())
    eligible = np.asarray(
        [index for index, group in enumerate(group_ids_np) if group_counts[int(group)] > 1],
        dtype=np.int64,
    )
    if query_indices is not None:
        requested = np.asarray(query_indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("query_indices must be one-dimensional")
        if len(requested) and (requested.min() < 0 or requested.max() >= len(array)):
            raise IndexError("query_indices contain an out-of-range row")
        if len(np.unique(requested)) != len(requested):
            raise ValueError("query_indices contain duplicates")
        eligible_set = set(eligible.tolist())
        selected = np.asarray([i for i in requested.tolist() if i in eligible_set], dtype=np.int64)
    else:
        selected = eligible
    if len(selected) == 0:
        raise ValueError("no selected query has a non-self relevant candidate")

    resolved_device = _resolve_device(device)
    if resolved_device.type == "cpu" and num_threads is not None:
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)

    embeddings_tensor = torch.from_numpy(array).to(resolved_device)
    embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
    group_ids = torch.from_numpy(group_ids_np).to(resolved_device)
    max_k = min(max(cutoffs), len(array) - 1)

    sums = {"mrr": 0.0}
    for k in cutoffs:
        sums[f"recall@{k}"] = 0.0
        sums[f"ndcg@{k}"] = 0.0

    start_time = time.perf_counter()
    query_batches = range(0, len(selected), query_batch_size)
    if progress:
        query_batches = tqdm(
            query_batches,
            desc="Evaluating query batches",
            unit="batch",
            dynamic_ncols=True,
        )
    with torch.inference_mode():
        for query_start in query_batches:
            batch_np = selected[query_start : query_start + query_batch_size]
            batch = torch.from_numpy(batch_np).to(resolved_device)
            queries = embeddings_tensor[batch]
            batch_groups_np = group_ids_np[batch_np]
            positive_lists = [
                [index for index in group_members[int(group)] if index != int(query)]
                for group, query in zip(batch_groups_np, batch_np)
            ]
            total_relevant = torch.as_tensor(
                [len(indices) for indices in positive_lists],
                dtype=torch.long,
                device=resolved_device,
            )
            best_positive_scores = _positive_score_thresholds(
                embeddings_tensor, batch, positive_lists
            )

            outranking_counts = torch.zeros(len(batch), dtype=torch.long, device=resolved_device)
            top_values = torch.empty((len(batch), 0), device=resolved_device)
            top_indices = torch.empty(
                (len(batch), 0), dtype=torch.long, device=resolved_device
            )

            for candidate_start in range(0, len(array), candidate_block_size):
                candidate_end = min(candidate_start + candidate_block_size, len(array))
                scores = queries @ embeddings_tensor[candidate_start:candidate_end].T

                in_block = (batch >= candidate_start) & (batch < candidate_end)
                if in_block.any():
                    rows = torch.nonzero(in_block, as_tuple=False).squeeze(1)
                    columns = batch[rows] - candidate_start
                    scores[rows, columns] = -torch.inf

                outranking_counts += torch.sum(
                    scores > best_positive_scores.unsqueeze(1), dim=1
                )

                block_k = min(max_k, scores.shape[1])
                block_values, block_local_indices = torch.topk(scores, block_k, dim=1)
                block_indices = block_local_indices + candidate_start
                merged_values = torch.cat((top_values, block_values), dim=1)
                merged_indices = torch.cat((top_indices, block_indices), dim=1)
                keep_k = min(max_k, merged_values.shape[1])
                top_values, positions = torch.topk(merged_values, keep_k, dim=1)
                top_indices = torch.gather(merged_indices, 1, positions)

            reciprocal_ranks = 1.0 / (outranking_counts.float() + 1.0)
            sums["mrr"] += reciprocal_ranks.sum().item()
            ranked_relevance = group_ids[top_indices] == group_ids[batch].unsqueeze(1)

            for k in cutoffs:
                effective_k = min(k, ranked_relevance.shape[1])
                hits = ranked_relevance[:, :effective_k].sum(dim=1)
                recall = hits.float() / total_relevant.float()
                sums[f"recall@{k}"] += recall.sum().item()

                dcg = _dcg_from_binary_relevance(ranked_relevance, effective_k)
                positions = torch.arange(
                    1, effective_k + 1, device=resolved_device, dtype=torch.long
                ).unsqueeze(0)
                ideal_relevance = positions <= total_relevant.unsqueeze(1)
                idcg = _dcg_from_binary_relevance(ideal_relevance, effective_k)
                sums[f"ndcg@{k}"] += (dcg / idcg).sum().item()

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed = time.perf_counter() - start_time
    metrics = {name: value / len(selected) for name, value in sums.items()}
    mean_positives = float(np.mean([group_counts[int(group_ids_np[i])] - 1 for i in eligible]))
    return RankingEvaluation(
        metrics=metrics,
        total_items=len(array),
        eligible_queries=len(eligible),
        evaluated_queries=len(selected),
        excluded_singletons=len(array) - len(eligible),
        mean_positives_per_query=mean_positives,
        candidate_pool_size=len(array) - 1,
        elapsed_seconds=elapsed,
    )
