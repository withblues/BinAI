import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import time
from datasets import load_from_disk
import json

def spearman_corr_batch(x, y):
    """
    Computes Spearman's rank correlation coefficient for a batch of tensors on the GPU.
    Args:
        x (torch.Tensor): A [batch_size, num_docs] tensor.
        y (torch.Tensor): A [batch_size, num_docs] tensor.
    Returns:
        (torch.Tensor) A [batch_size] tensor of Spearman's Rho scores.
    """
    # Get the ranks of the data along the document dimension.
    # torch.argsort twice is the standard way to get ranks.
    x_rank = torch.argsort(torch.argsort(x, dim=1)).float()
    y_rank = torch.argsort(torch.argsort(y, dim=1)).float()
    
    # Now, calculate Pearson correlation on the ranks.
    # Center the ranks (subtract the mean)
    x_rank_mean = torch.mean(x_rank, dim=1, keepdim=True)
    y_rank_mean = torch.mean(y_rank, dim=1, keepdim=True)
    x_centered = x_rank - x_rank_mean
    y_centered = y_rank - y_rank_mean
    
    # Calculate covariance and standard deviations
    covariance = torch.sum(x_centered * y_centered, dim=1)
    x_std_dev = torch.sqrt(torch.sum(x_centered**2, dim=1))
    y_std_dev = torch.sqrt(torch.sum(y_centered**2, dim=1))
    
    # Calculate the correlation coefficient
    # Add a small epsilon for numerical stability to avoid division by zero
    correlation = covariance / (x_std_dev * y_std_dev + 1e-12)
    
    return correlation

# This denominator is constant for a given k, so we compute it once on the GPU
# and reuse it. This is a key optimization for batched NDCG.
DISCOUNT_FACTORS_CACHE = {}

def dcg_batch(relevances_batch, k):
    """
    Computes Discounted Cumulative Gain for a batch of relevance scores.
    Args:
        relevances_batch: (torch.Tensor) A [batch_size, num_docs] tensor of relevance scores.
        k: (int) The cutoff value.
    Returns:
        (torch.Tensor) A [batch_size] tensor of DCG@k scores.
    """
    device = relevances_batch.device
    
    # Get the top-k relevance scores
    relevances_batch_k = relevances_batch[:, :k]
    
    # Pre-compute or retrieve discount factors from cache
    if k not in DISCOUNT_FACTORS_CACHE or DISCOUNT_FACTORS_CACHE[k].device != device:
        denominators = torch.log2(torch.arange(2, k + 2, device=device))
        DISCOUNT_FACTORS_CACHE[k] = 1.0 / denominators
        
    discount_factors = DISCOUNT_FACTORS_CACHE[k]
    
    # 2^relevance - 1
    numerators = torch.pow(2, relevances_batch_k) - 1
    
    # Sum the discounted gains for each item in the batch
    return torch.sum(numerators * discount_factors, dim=1)

def ndcg_at_k_batch(y_true_batch, y_score_batch, k):
    """
    Computes Normalized DCG for a batch of scores and ground truths.
    Args:
        y_true_batch: (torch.Tensor) A [batch_size, num_docs] tensor of ground truth relevance.
        y_score_batch: (torch.Tensor) A [batch_size, num_docs] tensor of predicted scores.
        k: (int) The cutoff value.
    Returns:
        (torch.Tensor) A [batch_size] tensor of NDCG@k scores.
    """
    # Get the ranking of documents based on predicted scores
    predicted_order = torch.argsort(y_score_batch, dim=1, descending=True)
    # Reorder the true relevance scores according to the prediction
    ranked_relevance = torch.gather(y_true_batch, 1, predicted_order)
    
    # Calculate the DCG of the prediction
    predicted_dcg = dcg_batch(ranked_relevance, k)
    
    # Get the ideal ranking based on true relevance
    ideal_order = torch.argsort(y_true_batch, dim=1, descending=True)
    # Reorder the true relevance scores for the ideal case
    ideal_relevance = torch.gather(y_true_batch, 1, ideal_order)
    
    # Calculate the DCG of the ideal ranking
    ideal_dcg = dcg_batch(ideal_relevance, k)
    
    # Avoid division by zero
    return torch.where(ideal_dcg > 0, predicted_dcg / ideal_dcg, torch.tensor(0.0, device=y_true_batch.device))

def mrr_batch(ranked_relevance_batch):
    hits_mask = ranked_relevance_batch > 0
    # set irrelevant positions to large index
    indices = torch.arange(ranked_relevance_batch.size(1), device=ranked_relevance_batch.device).unsqueeze(0).expand_as(ranked_relevance_batch)
    ranks = torch.where(hits_mask, indices + 1, torch.full_like(indices, ranked_relevance_batch.size(1) + 1))
    first_hit_ranks, _ = ranks.min(dim=1)
    reciprocal_ranks = torch.where(first_hit_ranks <= ranked_relevance_batch.size(1),
                                   1.0 / first_hit_ranks.float(),
                                   torch.zeros_like(first_hit_ranks, dtype=torch.float))
    return reciprocal_ranks


def precision_at_k_batch(ranked_relevance_batch, k):
    """
    Computes Precision@k for a batch of ranked relevance scores.
    """
    # Count the number of relevant documents in the top-k
    hits_at_k = torch.sum(ranked_relevance_batch[:, :k] > 0, dim=1)
    return hits_at_k.float() / k

def recall_at_k_batch(ranked_relevance_batch, k):
    """
    Computes Recall@k for a batch of ranked relevance scores.

    Args:
        ranked_relevance_batch (torch.Tensor): [batch_size, num_docs] relevance scores
                                               already sorted by predicted rank (desc).
        k (int): cutoff.
    Returns:
        torch.Tensor: [batch_size] tensor of Recall@k values.
    """
    device = ranked_relevance_batch.device
    
    # Relevant docs in top-k
    hits_at_k = torch.sum(ranked_relevance_batch[:, :k] > 0, dim=1)
    
    # Total relevant docs
    total_relevant = torch.sum(ranked_relevance_batch > 0, dim=1)
    
    # Safe divide: recall = hits_at_k / total_relevant
    recall = torch.where(
        total_relevant > 0,
        hits_at_k.float() / total_relevant.float(),
        torch.zeros_like(total_relevant, dtype=torch.float, device=device)
    )
    return recall


def process_finished_batch(sim_scores_gpu, batch_info, all_ids, idx_to_gt_key_map, gt_lookup, metrics_results, similarity_matrix_mmap, K_VALUES, num_rows, device, student_eval):
    """
    This function contains all the processing logic for a batch whose GPU computation is complete.
    """
    batch_start, batch_end, batch_size = batch_info

    # --- 1. METRICS CALCULATION (GPU-accelerated) ---
    # Build y_true batch on CPU (this is fast)
    y_true_batch_cpu = np.zeros((batch_size, num_rows), dtype=np.float32)
    for j in range(batch_size):
        anchor_idx = batch_start + j
        anchor_key = idx_to_gt_key_map.get(anchor_idx)
        if anchor_key and anchor_key in gt_lookup:
            y_true_batch_cpu[j, gt_lookup[anchor_key]] = 1
    
    # Move to GPU and calculate all metrics
    y_true_batch_gpu = torch.from_numpy(y_true_batch_cpu).to(device)
    
    predicted_order = torch.argsort(sim_scores_gpu, dim=1, descending=True)
    ranked_relevance = torch.gather(y_true_batch_gpu, 1, predicted_order)
    
    metrics_results['mrr'].extend(mrr_batch(ranked_relevance).cpu().tolist())
    for k in K_VALUES:
        metrics_results[f'precision@{k}'].extend(precision_at_k_batch(ranked_relevance, k).cpu().tolist())
        metrics_results[f'ndcg@{k}'].extend(ndcg_at_k_batch(y_true_batch_gpu, sim_scores_gpu, k).cpu().tolist())
        metrics_results[f'recall@{k}'].extend(recall_at_k_batch(ranked_relevance, k).cpu().tolist())

    # --- 2. SAVE TO DISK (CPU-side) ---
    mask = torch.ones_like(sim_scores_gpu, dtype=bool)
    row_indices = torch.arange(batch_size)
    col_indices = batch_start + row_indices
    mask[row_indices, col_indices] = False
    processed_batch = sim_scores_gpu[mask].view(current_batch_size, num_rows - 1)

    if not student_eval:
        similarity_matrix_mmap[batch_start:batch_end, :] = processed_batch.cpu().numpy()

    return processed_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a similarity matrix and save to np.memmap.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--method", required=True, help="Method name for the model (e.g., 'teacher_model')")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for GPU queries. Tune based on VRAM.")
    parser.add_argument("--model_name", default='clap', help="Model name for student-teacher evaluation.")
    parser.add_argument("--embeddings_dataset_path", type=str, required=True, help="Path to the pre-computed embeddings dataset.")
    parser.add_argument("--skip_teacher", action="store_true", help="Skip loading teacher embeddings and skip teacher comparison metrics.")
    args = parser.parse_args()

    # The method and init_suffix are now only for naming the output report
    method_name_with_suffix = args.method

    print(f'compute metrics with {method_name_with_suffix} on split {args.split}')
    IS_STUDENT_EVAL = args.method not in ['clap', 'deepseek', 'starcoder2', 'qwen' ,'llm4decompile', 'nova']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device {device}')
    K_VALUES = [1, 5, 10, 50, 100, 512, 1024]

    source_dataset_path = args.embeddings_dataset_path
    print(f'Loading dataset from {source_dataset_path}')
    source_dataset = load_from_disk(source_dataset_path)
    source_dataset.set_format("numpy", columns=['unique_id', 'embedding'])

    with open(os.path.join(args.data_dir, f"cross_{args.split}_split.json")) as f:
        indices = json.load(f)

    test_ids = set(indices["test"])

    # source_dataset = load_from_disk(os.path.join(args.data_dir, f'assembly_x64_1024_{args.method}'))
    # source_dataset = source_dataset.rename_column(f'{args.method}_embedding', 'embedding')
    # source_dataset = source_dataset.filter(lambda batch: [uid in test_ids for uid in batch["unique_id"]], batched=True, num_proc=16)
    # source_dataset.set_format("numpy", columns=['unique_id', 'embedding'])

    # load into ram
    all_data_np = source_dataset[:] 
    all_ids = all_data_np['unique_id']

    all_embeddings_np = np.ascontiguousarray(all_data_np['embedding'], dtype=np.float32)
    num_rows, dim = all_embeddings_np.shape
    print(f"Loaded {num_rows} embeddings of dimension {dim}.")
    test_ids_set = set(all_ids)

    if IS_STUDENT_EVAL and not args.skip_teacher:
        print(f"Loading pre-computed teacher scores for {args.model_name}")
        teacher_matrix_path = os.path.join(args.output_dir, 'inference/cosine_scores', args.split, f"{args.model_name}_similarity_matrix.mmap")
        teacher_ids_path = os.path.join(args.output_dir, 'inference/cosine_scores', args.split, f"{args.model_name}_ids.npy")

        teacher_ids = np.load(teacher_ids_path, allow_pickle=True)
        assert np.array_equal(all_ids, teacher_ids), "Mismatch between student and teacher IDs!"
        
        teacher_similarity_mmap = np.memmap(teacher_matrix_path, dtype='float32', mode='r', shape=(num_rows, num_rows - 1)) 

    # load grount truth dataset
    metadata_full_dataset = load_from_disk(os.path.join(args.data_dir, 'assembly_x64_1024_clap'))
    metadata_dataset = metadata_full_dataset.filter(
        lambda x: x['unique_id'] in test_ids_set,
        num_proc=16 # Use multiple processes for faster filtering
    )
    metadata_dataset.set_format(columns=['unique_id', 'binary_name', 'function_name'])

    # create mapping
    id_to_idx_map = {uid: i for i, uid in enumerate(all_ids)}
    
    #  GT map (key -> list of indices)
    idx_to_gt_key_map = {}
    for row in metadata_dataset:
        uid = row['unique_id']
        if uid in id_to_idx_map:
            idx = id_to_idx_map[uid]
            idx_to_gt_key_map[idx] = (row['binary_name'], row['function_name'])

    # GT lookup map (key -> list of indices)
    gt_lookup = {}
    for idx, key in idx_to_gt_key_map.items():
        if key not in gt_lookup:
            gt_lookup[key] = []
        gt_lookup[key].append(idx)

    # output path
    output_path_matrix = os.path.join(args.output_dir, 'inference/cosine_scores',  args.split, f"{method_name_with_suffix}_similarity_matrix.mmap")
    output_path_ids = os.path.join(args.output_dir, 'inference/cosine_scores',  args.split, f"{method_name_with_suffix}_ids.npy")
    os.makedirs(os.path.dirname(output_path_matrix), exist_ok=True)

    # allocate space
    if not IS_STUDENT_EVAL:
        print(f"Running in TEACHER mode. Scores will be saved to disk.")
        output_path_matrix = os.path.join(args.output_dir, 'inference/cosine_scores',  args.split, f"{method_name_with_suffix}_similarity_matrix.mmap")
        output_path_ids = os.path.join(args.output_dir, 'inference/cosine_scores',  args.split, f"{method_name_with_suffix}_ids.npy")
        os.makedirs(os.path.dirname(output_path_matrix), exist_ok=True)
        similarity_matrix_mmap = np.memmap(output_path_matrix, dtype='float32', mode='w+', shape=(num_rows, num_rows - 1))    
        np.save(output_path_ids, all_ids)
    else:
        print(f"Running in STUDENT mode. Scores will NOT be saved to disk.")
        similarity_matrix_mmap = None

    # setup torch 
    all_embeddings_gpu = torch.from_numpy(all_embeddings_np).to(device)
    all_embeddings_T_gpu = all_embeddings_gpu.T


    # dict for metrics
    metrics_results = {'mrr': []}
    if IS_STUDENT_EVAL:
        metrics_results['spearman'] = []

    for k in K_VALUES:
        metrics_results[f'ndcg@{k}'] = []
        metrics_results[f'precision@{k}'] = []
        metrics_results[f'recall@{k}'] = []

    start_time = time.time()
    prev_sim_scores_gpu = None
    prev_batch_info = None

    for i in tqdm(range(0, num_rows, args.batch_size), desc="Writing Similarity Batches"):
        batch_start, batch_end = i, min(i + args.batch_size, num_rows)
        current_batch_size = batch_end - batch_start

        query_batch_gpu = all_embeddings_gpu[batch_start:batch_end]
        sim_scores_gpu = torch.matmul(query_batch_gpu, all_embeddings_T_gpu)

        # ranked base metrics
        y_true_batch_cpu = np.zeros((current_batch_size, num_rows), dtype=np.float32)
        for j in range(current_batch_size):
            anchor_idx = batch_start + j
            anchor_key = idx_to_gt_key_map.get(anchor_idx)
            if anchor_key and anchor_key in gt_lookup:
                y_true_batch_cpu[j, gt_lookup[anchor_key]] = 1
        
        y_true_batch_gpu = torch.from_numpy(y_true_batch_cpu).to(device)
        predicted_order = torch.argsort(sim_scores_gpu, dim=1, descending=True)
        ranked_relevance = torch.gather(y_true_batch_gpu, 1, predicted_order)
        
        metrics_results['mrr'].extend(mrr_batch(ranked_relevance).cpu().tolist())
        for k in K_VALUES:
            metrics_results[f'precision@{k}'].extend(precision_at_k_batch(ranked_relevance, k).cpu().tolist())
            metrics_results[f'ndcg@{k}'].extend(ndcg_at_k_batch(y_true_batch_gpu, sim_scores_gpu, k).cpu().tolist())
            metrics_results[f'recall@{k}'].extend(recall_at_k_batch(ranked_relevance, k).cpu().tolist())

        # distillation metrics
        mask = torch.ones_like(sim_scores_gpu, dtype=torch.bool)
        row_indices = torch.arange(current_batch_size, device=device)
        col_indices = batch_start + row_indices
        mask[row_indices, col_indices] = False
        sim_scores_processed_gpu = sim_scores_gpu[mask].view(current_batch_size, num_rows - 1)

        if IS_STUDENT_EVAL:
            if not args.skip_teacher:
                # load teacher scores
                sim_scores_teacher_cpu = teacher_similarity_mmap[batch_start:batch_end, :].copy() 
                sim_scores_teacher_gpu = torch.from_numpy(sim_scores_teacher_cpu).to(device)
                
                # spearman
                spearman_scores = spearman_corr_batch(sim_scores_processed_gpu, sim_scores_teacher_gpu)
                metrics_results['spearman'].extend(spearman_scores.cpu().tolist())
        else: 
            # save to disk
            similarity_matrix_mmap[batch_start:batch_end, :] = sim_scores_processed_gpu.cpu().numpy()
        



    # flush changes to disk to ensure everything is saved
    if not IS_STUDENT_EVAL:
        similarity_matrix_mmap.flush()

    end_time = time.time()
    print(f"\nMatrix generation complete. Took {(end_time - start_time) / 60:.2f} minutes.")
    
    if metrics_results['mrr']:
        final_report = {"model_name": method_name_with_suffix}
        print("\n--- Teacher Performance Metrics ---")
        final_report['mrr'] = np.mean(metrics_results['mrr'])
        print(f"Teacher MRR: {final_report['mrr']:.4f}")

        if IS_STUDENT_EVAL and not args.skip_teacher:
            final_report['spearman'] = np.mean(metrics_results['spearman'])
            print(f"Teacher spearman: {final_report['spearman']:.4f}")

        for k in K_VALUES:
            final_report[f'ndcg@{k}'] = np.mean(metrics_results[f'ndcg@{k}'])
            final_report[f'precision@{k}'] = np.mean(metrics_results[f'precision@{k}'])
            final_report[f'recall@{k}'] = np.mean(metrics_results[f'recall@{k}']) # <-- ADD THIS LINE
            print(f"Teacher NDCG@{k}: {final_report[f'ndcg@{k}']:.4f}")
            print(f"Teacher Precision@{k}: {final_report[f'precision@{k}']:.4f}")
            print(f"Teacher Recall@{k}: {final_report[f'recall@{k}']:.4f}")

        report_path = os.path.join(args.output_dir, 'inference/metrics', args.split, args.model_name, f"{method_name_with_suffix}_metrics_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=4)
        print(f"\nTeacher metrics report saved to: {report_path}")