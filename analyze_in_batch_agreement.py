import argparse
import os
import numpy as np
import json
from datasets import load_from_disk
from tqdm import tqdm
import random
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--teacher_type", default='clap')
    parser.add_argument("--batch_size", type=int, default=128, help="Simulated batch size")
    parser.add_argument("--num_batches", type=int, default=1000, help="Number of batches to simulate")
    parser.add_argument("--structured_batch", action='store_true', help="If true, construct batches as [Anchors, Positives] like InBatchInfoNCECollator")
    args = parser.parse_args()

    print(f"Loading split {args.split}...")
    with open(os.path.join(args.data_dir, f"cross_{args.split}_split.json")) as f:
        indices = json.load(f)
    test_ids = set(indices["test"])

    print("Loading dataset...")
    dataset = load_from_disk(os.path.join(args.data_dir, f'assembly_x64_1024_{args.teacher_type}'))
    
    print("Filtering to test set...")
    test_dataset = dataset.filter(lambda x: x['unique_id'] in test_ids, num_proc=16)
    
    print("Formatting dataset...")
    test_dataset.set_format(columns=['unique_id', 'binary_name', 'function_name', f'{args.teacher_type}_embedding'])

    print("Loading all embeddings into RAM...")
    all_embeddings = np.array(test_dataset[f'{args.teacher_type}_embedding'])
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    all_embeddings = all_embeddings / norms
    
    print("Loading metadata...")
    binary_names = np.array(test_dataset['binary_name'])
    function_names = np.array(test_dataset['function_name'])
    
    num_items = len(all_embeddings)
    
    # Pre-compute groups if using structured batches
    valid_positive_groups = {}
    if args.structured_batch:
        print("Building lookup structures for structured batches...")
        gt_groups = defaultdict(list)
        for i in tqdm(range(num_items)):
            key = (binary_names[i], function_names[i])
            gt_groups[key].append(i)
        valid_positive_groups = {k: v for k, v in gt_groups.items() if len(v) > 1}
        if len(valid_positive_groups) < args.batch_size // 2:
            print("Not enough positive pairs in dataset to form a full batch.")
            return

    auroc_scores = []
    pearson_scores = []
    
    print(f"Simulating {args.num_batches} batches of size {args.batch_size}...")
    group_keys = list(valid_positive_groups.keys()) if valid_positive_groups else []

    for _ in tqdm(range(args.num_batches)):
        if args.structured_batch:
            # Sample B/2 pairs
            half_b = args.batch_size // 2
            selected_keys = random.sample(group_keys, half_b)
            anchors = []
            positives = []
            for k in selected_keys:
                i1, i2 = random.sample(valid_positive_groups[k], 2)
                anchors.append(i1)
                positives.append(i2)
            batch_indices = anchors + positives
        else:
            # Sample completely randomly
            batch_indices = random.sample(range(num_items), args.batch_size)
        
        batch_embeddings = all_embeddings[batch_indices]
        batch_bins = binary_names[batch_indices]
        batch_funcs = function_names[batch_indices]
        
        # Compute BxB similarities
        sim_matrix = np.dot(batch_embeddings, batch_embeddings.T)
        
        # Compute ground truth
        same_bin = (batch_bins[:, None] == batch_bins[None, :])
        same_func = (batch_funcs[:, None] == batch_funcs[None, :])
        gt_matrix = (same_bin & same_func).astype(int)
        
        # Mask out diagonal (self-similarity)
        np.fill_diagonal(gt_matrix, -1)
        
        # Flatten and filter out diagonal
        sim_flat = sim_matrix[gt_matrix != -1]
        gt_flat = gt_matrix[gt_matrix != -1]
        
        # If there are no positive pairs in this random batch, AUROC is undefined.
        if np.sum(gt_flat) == 0 or np.sum(gt_flat) == len(gt_flat):
            continue
            
        # AUROC
        auroc = roc_auc_score(gt_flat, sim_flat)
        auroc_scores.append(auroc)
        
        # Pearson
        pearson, _ = pearsonr(sim_flat, gt_flat)
        pearson_scores.append(pearson)
        
    if not auroc_scores:
        print("Could not compute metrics. Random batches contained no positive pairs.")
        print("Try using --structured_batch or increasing the batch size.")
        return
        
    mean_auroc = np.mean(auroc_scores)
    mean_pearson = np.mean(pearson_scores)
    
    print("\n--- In-Batch Label Agreement Analysis ---")
    print(f"Teacher: {args.teacher_type}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Structured Batches: {args.structured_batch}")
    print(f"Number of valid batches analyzed: {len(auroc_scores)}")
    print(f"Mean AUROC: {mean_auroc:.4f}")
    print(f"Mean Pearson Correlation: {mean_pearson:.4f}")

if __name__ == '__main__':
    main()
