import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import numpy as np
import argparse
from datasets import load_from_disk, Dataset
import gc
from tqdm import tqdm
import faiss
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leak-free hard negative pool with FAISS and cosine scores for Knowledge Distillation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the original dataset with embeddings.")
    parser.add_argument("--splits_dir", type=str, required=True, help="Directory with JSON files defining train/val splits.")
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_faiss_kd", help="Directory to save the final combined dataset.")
    parser.add_argument("--split", type=str, default='project')
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=10, help="Total number of targets to generate for each anchor.")
    parser.add_argument("--num_candidates_faiss", type=int, default=512, help="Number of candidates to retrieve from FAISS.")
    parser.add_argument("--num_hard_positives", type=int, default=1, help="Number of targets to sample from the high-similarity pool.")
    parser.add_argument("--num_hard_negatives", type=int, default=5, help="Number of targets to sample from the mid-similarity pool.")
    parser.add_argument("--num_random_negatives", type=int, default=4, help="Number of targets to sample randomly.")
    parser.add_argument("--hard_positive_threshold", type=float, default=0.7, help="Min score for a candidate to be a 'hard positive'.")
    parser.add_argument("--hard_negative_max_score", type=float, default=0.7, help="Max score for a candidate to be a 'hard negative'.")
    parser.add_argument("--hard_negative_min_score", type=float, default=0.2, help="Min score for a candidate to be a 'hard negative'.")
    args = parser.parse_args()

    split_path = f'cross_{args.split}_split'
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    assert (args.num_hard_positives + args.num_hard_negatives + args.num_random_negatives == args.top_k), "The sum of positive/negative counts must equal top_k."

    # --- Load full dataset to be used as a master source ---
    print("Loading full dataset...")
    dataset = load_from_disk(args.data_dir)

    # --- Load split definitions ---
    split_file_path = os.path.join(args.splits_dir, f"{split_path}.json")
    with open(split_file_path, 'r') as f:
        split_ids_by_group = json.load(f)

    # --- Global lists to accumulate data from all splits ---
    final_unique_ids, final_split_column, final_target_ids, final_cosine_scores = [], [], [], []

    for data_split in ["train", "val"]:
        if data_split not in split_ids_by_group: continue
        print(f"\n--- Hard mining for '{data_split}' split ---")

        # --- Create the isolated universe for this split using .filter() ---
        split_ids_set = set(split_ids_by_group[data_split])
        split_dataset = dataset.filter(lambda x: x['unique_id'] in split_ids_set, num_proc=32)
        if len(split_dataset) == 0:
            print(f"No data for split '{data_split}'. Skipping.")
            continue

        # --- THE KEY FIX for the KeyError ---
        # Set the format to include ALL columns you need to access before slicing.
        split_dataset.set_format("numpy", columns=["unique_id", "clap_embedding"])
        
        # Now you can safely access both columns from the sliced data
        split_data_np = split_dataset[:]
        split_embeddings = split_data_np["clap_embedding"].astype('float32')
        split_ids = list(split_data_np['unique_id'])
        # ------------------------------------
        
        split_num_rows, dim = split_embeddings.shape
        print(f"Created isolated dataset for '{data_split}' with {split_num_rows} functions.")

        # --- Build FAISS index only for this split ---
        nlist = max(1, min(4096, split_num_rows // 100))
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        train_sample_size = min(split_num_rows, 256 * 1024)
        random_local_indices = np.random.choice(split_num_rows, size=train_sample_size, replace=False)
        cpu_index.train(split_embeddings[random_local_indices])
        cpu_index.add(split_embeddings)
        
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.nprobe = 32

        # --- Start Mining ---
        anchor_indices_in_split = np.arange(split_num_rows)

        for start in tqdm(range(0, len(anchor_indices_in_split), args.batch_size), desc=f"{data_split} batches"):
            end = min(start + args.batch_size, len(anchor_indices_in_split))
            batch_local_indices = anchor_indices_in_split[start:end]
            batch_embeddings = split_embeddings[batch_local_indices]

            scores, indices = gpu_index.search(batch_embeddings, args.num_candidates_faiss + 1)

            for i, local_idx in enumerate(batch_local_indices):
                anchor_id = split_ids[local_idx]
                
                # --- Restore the full KD sampling logic ---
                candidate_indices = indices[i]
                candidate_scores = scores[i]
                mask_self = candidate_indices == local_idx

                pos_mask = (candidate_scores >= args.hard_positive_threshold) & ~mask_self
                hard_neg_mask = (candidate_scores >= args.hard_negative_min_score) & (candidate_scores < args.hard_negative_max_score) & ~mask_self

                pos_candidates = candidate_indices[pos_mask]
                hard_neg_candidates = candidate_indices[hard_neg_mask]

                selected_targets = []

                # Sample hard positives
                if len(pos_candidates) > 0:
                    num_to_add = min(len(pos_candidates), args.num_hard_positives)
                    selected_targets.extend(np.random.choice(pos_candidates, num_to_add, replace=False).tolist())

                # Sample hard negatives
                if len(hard_neg_candidates) > 0:
                    num_to_add = min(len(hard_neg_candidates), args.num_hard_negatives)
                    selected_targets.extend(np.random.choice(hard_neg_candidates, num_to_add, replace=False).tolist())

                # Sample random negatives (from split only)
                num_random_to_add = args.top_k - len(selected_targets)
                if num_random_to_add > 0:
                    forbidden = set(selected_targets)
                    forbidden.add(local_idx)
                    rand_candidates = []
                    while len(rand_candidates) < num_random_to_add:
                        r = random.randint(0, split_num_rows - 1)
                        if r not in forbidden:
                            rand_candidates.append(r)
                            forbidden.add(r)
                    selected_targets.extend(rand_candidates)
                
                # Ensure we have exactly top_k targets, fall back to just random if needed
                if len(selected_targets) < args.top_k:
                    # This fallback is rare but robust
                    selected_targets = []
                    forbidden = {local_idx}
                    while len(selected_targets) < args.top_k:
                        r = random.randint(0, split_num_rows - 1)
                        if r not in forbidden:
                            selected_targets.append(r)
                            forbidden.add(r)
                
                selected_targets = selected_targets[:args.top_k]

                # Map local indices back to global IDs and compute cosine scores
                target_ids = [split_ids[idx] for idx in selected_targets]
                target_embeddings = split_embeddings[selected_targets]
                target_scores = (split_embeddings[local_idx] @ target_embeddings.T).tolist()
                
                # Append to the global lists
                final_unique_ids.append(anchor_id)
                final_split_column.append(data_split)
                final_target_ids.append(target_ids)
                final_cosine_scores.append(target_scores)
    
    # --- Build and save the COMBINED dataset ---
    if not final_unique_ids:
        print("No valid pairs found.")
    else:
        final_ds = Dataset.from_dict({
            "unique_id": final_unique_ids,
            "split": final_split_column,
            "target_ids": final_target_ids,
            "cosine_scores": final_cosine_scores
        })
        # Save a single combined file
        output_path = os.path.join(args.output_dir, split_path)
        print(f"Saving combined dataset to {output_path}")
        final_ds.save_to_disk(output_path)

    print("\nAll splits processed successfully!")