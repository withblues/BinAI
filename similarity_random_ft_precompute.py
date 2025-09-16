import os
import json
import numpy as np
import argparse
from datasets import load_from_disk, Dataset
import gc
from tqdm import tqdm
import random

# This script generates the baseline dataset for the Fine-Tuning (FT) paradigm.
# For each anchor, it guarantees one Ground Truth positive and fills the rest
# of the target list with purely random negatives.

if __name__ == "__main__":
    # --- Arguments ---
    parser = argparse.ArgumentParser(description="Generate a random baseline dataset for Fine-Tuning.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the original dataset with embeddings and metadata.")
    parser.add_argument("--splits_dir", type=str, required=True, help="Directory with JSON files defining train/val splits.")
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_random_ft")
    parser.add_argument("--batch_size", type=int, default=8192, help="Number of anchors to process per loop.")
    parser.add_argument("--top_k", type=int, default=10, help="Total number of targets (1 positive + k-1 negatives).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--debug_subset_size", type=int, default=None, 
                        help="Run on a small subset of N anchors for debugging.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    num_negatives_to_sample = args.top_k - 1

    random.seed(42)
    np.random.seed(42)

    # --- Set Seed for Reproducibility ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # --- 1. Load data and build ground truth lookup table ---
    print("Loading full dataset and building lookups...")
    original_dataset = load_from_disk(args.data_dir)
    if "clap_embedding" in original_dataset.column_names:
        original_dataset = original_dataset.rename_column("clap_embedding", "embedding")
    original_dataset.set_format("numpy", columns=['unique_id', 'binary_name', 'function_name', 'embedding'])
    all_data_np = original_dataset[:]
    
    embeddings = all_data_np['embedding'].astype('float32')
    all_ids = all_data_np['unique_id'].tolist()
    num_rows, dim = embeddings.shape
    id_to_index = {uid: i for i, uid in enumerate(all_ids)}

    # Build the lookup table to find GT positives efficiently
    gt_lookup = {}
    for uid, b_name, f_name in tqdm(zip(all_data_np['unique_id'], all_data_np['binary_name'], all_data_np['function_name']), total=len(all_data_np['unique_id']), desc="Building GT lookup"):
        uid_int, b_name_str, f_name_str = int(uid), str(b_name), str(f_name)
        if b_name_str not in gt_lookup: gt_lookup[b_name_str] = {}
        if f_name_str not in gt_lookup[b_name_str]: gt_lookup[b_name_str][f_name_str] = []
        gt_lookup[b_name_str][f_name_str].append(uid_int)
    
    del all_data_np
    print("Lookups built.")

    # --- 2. Process each split JSON file ---
    for split_file in os.listdir(args.splits_dir):
        if not split_file.endswith(".json"): continue
        split_name = os.path.splitext(split_file)[0]
        print(f"\nProcessing split file: {split_name}")
        json_path = os.path.join(args.splits_dir, split_file)
        if os.path.getsize(json_path) == 0: continue
        with open(json_path, 'r') as f:
            split_ids_by_group = json.load(f)

        final_anchor_ids, final_target_ids, final_cosine_scores, final_splits = [], [], [], []

        for data_split in ["train", "val"]:
            if data_split not in split_ids_by_group: continue
            print(f"--- Generating FT random baseline for '{data_split}' set ---")
            split_ids = split_ids_by_group[data_split]
            
            if args.debug_subset_size is not None:
                print(f"!!! RUNNING IN DEBUG MODE ON A SUBSET OF {args.debug_subset_size} ANCHORS !!!")
                split_ids = split_ids[:args.debug_subset_size]

            # Process in batches for memory efficiency
            for start in tqdm(range(0, len(split_ids), args.batch_size), desc=f"Processing {data_split} batches"):
                end = min(start + args.batch_size, len(split_ids))
                batch_anchor_ids = split_ids[start:end]
                
                for anchor_id in batch_anchor_ids:
                    anchor_idx = id_to_index[anchor_id]
                    anchor_meta = {'binary_name': original_dataset[anchor_idx]['binary_name'], 'function_name': original_dataset[anchor_idx]['function_name']}

                    # --- CORE LOGIC: Find one GT Positive ---
                    all_gt_positives = gt_lookup[str(anchor_meta['binary_name'])][str(anchor_meta['function_name'])]
                    guaranteed_positive_id = None
                    valid_gt_positives = [pid for pid in all_gt_positives if pid != anchor_id]
                    if not valid_gt_positives:
                        continue
                    guaranteed_positive_id = random.choice(valid_gt_positives)
                    
                    # If an anchor has no other GT positives in the dataset, we must skip it.
                    if guaranteed_positive_id is None:
                        continue
                    
                    guaranteed_positive_idx = id_to_index[guaranteed_positive_id]
                    
                    # --- CORE LOGIC: Sample k-1 Random Negatives ---
                    sampled_negatives = []
                    forbidden = {anchor_idx, guaranteed_positive_idx}
                    
                    while len(sampled_negatives) < num_negatives_to_sample:
                        rand_idx = random.randint(0, num_rows - 1)
                        if rand_idx not in forbidden:
                            sampled_negatives.append(rand_idx)
                            forbidden.add(rand_idx) # Avoid picking the same random sample twice
                    
                    # --- Construct final target list (Positive First) ---
                    final_target_indices = [guaranteed_positive_idx] + sampled_negatives
                    
                    # Calculate scores for plotting and analysis
                    target_scores = (embeddings[anchor_idx] @ embeddings[final_target_indices].T).tolist()
                    target_ids = [all_ids[idx] for idx in final_target_indices]
                    
                    final_anchor_ids.append(anchor_id)
                    final_target_ids.append(target_ids)
                    final_cosine_scores.append(target_scores)
                    final_splits.append(data_split)

            gc.collect()
        
        # --- 3. Build and save the final dataset ---
        if not final_anchor_ids: continue
        final_ds = Dataset.from_dict({
            "unique_id": final_anchor_ids,
            "target_ids": final_target_ids,
            "cosine_scores": final_cosine_scores,
            "split": final_splits
        })
        output_path = os.path.join(args.output_dir, split_name)
        print(f"Saving FT random baseline dataset to {output_path}")
        final_ds.save_to_disk(output_path)
        del final_ds
        gc.collect()

    print("\nAll splits processed successfully!")