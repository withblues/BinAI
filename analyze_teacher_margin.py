import argparse
import os
import torch
import numpy as np
import json
from datasets import load_from_disk
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--teacher_type", default='clap')
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of positive and negative pairs to sample")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading split {args.split}...")
    with open(os.path.join(args.data_dir, f"cross_{args.split}_split.json")) as f:
        indices = json.load(f)
    test_ids = set(indices["test"])

    print("Loading dataset...")
    dataset = load_from_disk(os.path.join(args.data_dir, f'assembly_x64_1024_{args.teacher_type}'))
    
    print("Filtering to test set...")
    test_dataset = dataset.filter(lambda x: x['unique_id'] in test_ids, num_proc=16)
    
    # We only need embeddings and ground truth keys
    print("Formatting dataset...")
    test_dataset.set_format(columns=['unique_id', 'binary_name', 'function_name', f'{args.teacher_type}_embedding'])

    print("Building lookup structures...")
    # Group by ground truth key (binary_name, function_name)
    gt_groups = defaultdict(list)
    all_indices = []
    
    for i, row in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        key = (row['binary_name'], row['function_name'])
        gt_groups[key].append(i)
        all_indices.append(i)
        
    print(f"Found {len(gt_groups)} unique (binary, function) groups.")
    
    # Identify groups that have at least 2 items (so we can form a positive pair)
    valid_positive_groups = {k: v for k, v in gt_groups.items() if len(v) > 1}
    print(f"Found {len(valid_positive_groups)} groups with at least 2 items.")

    if len(valid_positive_groups) == 0:
        print("No positive pairs found! Exiting.")
        return

    # Sample positive pairs
    print(f"Sampling up to {args.num_samples} positive pairs...")
    positive_pairs = []
    group_keys = list(valid_positive_groups.keys())
    
    while len(positive_pairs) < args.num_samples:
        k = random.choice(group_keys)
        group_items = valid_positive_groups[k]
        i1, i2 = random.sample(group_items, 2)
        positive_pairs.append((i1, i2))
        
    # Sample negative pairs
    print(f"Sampling {args.num_samples} negative pairs...")
    negative_pairs = []
    all_group_keys = list(gt_groups.keys())
    while len(negative_pairs) < args.num_samples:
        k1, k2 = random.sample(all_group_keys, 2) # Ensure they come from different groups
        i1 = random.choice(gt_groups[k1])
        i2 = random.choice(gt_groups[k2])
        negative_pairs.append((i1, i2))

    print("Fetching embeddings and computing similarities...")
    
    # Pre-extract embeddings to a numpy array for fast indexing
    print("Loading all embeddings into RAM...")
    all_embeddings = np.array(test_dataset[f'{args.teacher_type}_embedding'])
    
    # Normalize them to compute cosine similarity easily via dot product
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    all_embeddings = all_embeddings / norms
    
    print("Computing positive similarities...")
    pos_sims = []
    for i1, i2 in tqdm(positive_pairs):
        sim = np.dot(all_embeddings[i1], all_embeddings[i2])
        pos_sims.append(sim)
        
    print("Computing negative similarities...")
    neg_sims = []
    for i1, i2 in tqdm(negative_pairs):
        sim = np.dot(all_embeddings[i1], all_embeddings[i2])
        neg_sims.append(sim)
        
    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)
    
    # Calculate statistics
    pos_mean = np.mean(pos_sims)
    pos_std = np.std(pos_sims)
    neg_mean = np.mean(neg_sims)
    neg_std = np.std(neg_sims)
    
    print("\n--- Teacher Margin Analysis ---")
    print(f"Positive Pairs (N={len(pos_sims)}): Mean = {pos_mean:.4f}, Std = {pos_std:.4f}")
    print(f"Negative Pairs (N={len(neg_sims)}): Mean = {neg_mean:.4f}, Std = {neg_std:.4f}")
    print(f"Margin (Pos Mean - Neg Mean): {pos_mean - neg_mean:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(pos_sims, bins=100, alpha=0.5, label=f'Positive (Mean: {pos_mean:.2f})', density=True, color='blue')
    plt.hist(neg_sims, bins=100, alpha=0.5, label=f'Negative (Mean: {neg_mean:.2f})', density=True, color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title(f'Teacher Embedding Space Margin ({args.teacher_type})')
    plt.legend(loc='upper left')
    
    plot_path = os.path.join(args.output_dir, f'teacher_{args.teacher_type}_margin.png')
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")

if __name__ == '__main__':
    main()
