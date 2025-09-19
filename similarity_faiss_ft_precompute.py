import os
import json
import numpy as np
import argparse
from datasets import load_from_disk, Dataset
import gc
from tqdm import tqdm
import faiss
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid FAISS dataset generation for fine-tuning with score-based hard negative mining.")
    # --- Arguments from BOTH scripts ---
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_faiss_ft")
    parser.add_argument("--split", type=str, default='project')
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=10, help="Total number of targets to generate for each anchor.")
    parser.add_argument("--num_candidates_faiss", type=int, default=512, help="Number of candidates to retrieve from FAISS.")
    # Quotas for negative sampling
    parser.add_argument("--num_hard_negatives", type=int, default=5, help="Number of negatives to sample from the mid-similarity pool.")
    parser.add_argument("--num_random_negatives", type=int, default=4, help="Number of negatives to sample randomly or from the low-similarity pool.")
    # Score thresholds
    parser.add_argument("--hard_positive_threshold", type=float, default=0.7, help="Min score for a candidate to be a 'hard positive'.")
    parser.add_argument("--hard_negative_max_score", type=float, default=0.7, help="Max score for a candidate to be a 'hard negative'.")
    parser.add_argument("--hard_negative_min_score", type=float, default=0.2, help="Min score for a candidate to be a 'hard negative'.")
    args = parser.parse_args()

    # We need exactly 1 positive, so the sum of negatives must be top_k - 1
    assert (args.num_hard_negatives + args.num_random_negatives == args.top_k - 1), "The sum of negative counts must equal top_k - 1."

    split_path = f'cross_{args.split}_split'
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Setup: Load full dataset once ---
    print("Loading full dataset to prepare splits...")
    original_dataset = load_from_disk(args.data_dir)
    split_file_path = os.path.join(args.splits_dir, f"{split_path}.json")
    with open(split_file_path, 'r') as f:
        split_ids_by_group = json.load(f)

    # --- Master Loop: Process each split IN ISOLATION ---
    final_anchor_ids, final_positive_ids, final_negative_ids = [], [], []
    final_positive_scores, final_negative_scores, final_splits = [], [], []

    for data_split in ["train", "val"]:
        print(f"\n{'='*20} PROCESSING SPLIT: {data_split.upper()} {'='*20}")
        if data_split not in split_ids_by_group: continue

        # --- Step 1: Create the isolated universe for this split ---
        split_ids_set = set(split_ids_by_group[data_split])
        split_dataset = original_dataset.filter(lambda x: x['unique_id'] in split_ids_set, num_proc=32)
        if len(split_dataset) == 0: continue
        
        # --- Bulk data extraction ---
        split_dataset.set_format("numpy", columns=['unique_id', 'binary_name', 'function_name', 'clap_embedding'])
        split_data_np = split_dataset[:]
        split_embeddings = split_data_np['clap_embedding'].astype('float32')
        split_all_ids = split_data_np['unique_id'].tolist()
        split_all_binary_names = split_data_np['binary_name'].tolist()
        split_all_function_names = split_data_np['function_name'].tolist()
        split_num_rows, dim = split_embeddings.shape
        split_id_to_index = {uid: i for i, uid in enumerate(split_all_ids)}
        
        # --- Build local GT lookup ---
        gt_lookup_split = {}
        for i in tqdm(range(split_num_rows), desc=f"Building {data_split} GT lookup"):
            uid_int, b_name_str, f_name_str = int(split_all_ids[i]), str(split_all_binary_names[i]), str(split_all_function_names[i])
            if b_name_str not in gt_lookup_split: gt_lookup_split[b_name_str] = {}
            if f_name_str not in gt_lookup_split[b_name_str]: gt_lookup_split[b_name_str][f_name_str] = []
            gt_lookup_split[b_name_str][f_name_str].append(uid_int)

        # --- Pre-filter for valid anchors ---
        valid_anchor_indices = []
        for idx in range(split_num_rows):
             meta = {'binary_name': split_all_binary_names[idx], 'function_name': split_all_function_names[idx]}
             all_gt = gt_lookup_split[meta['binary_name']][meta['function_name']]
             if len(all_gt) > 1:
                 valid_anchor_indices.append(idx)
        valid_anchor_indices = np.array(valid_anchor_indices)
        print(f"Found {len(valid_anchor_indices)} valid anchors with at least one positive pair.")

        # --- Build FAISS index ---
        nlist = max(1, min(4096, split_num_rows // 100))
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        cpu_index.train(split_embeddings[np.random.choice(split_num_rows, size=min(split_num_rows, 256*1024), replace=False)])
        cpu_index.add(split_embeddings)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.nprobe = 32

        # --- Step 2: Process ONLY the valid anchors ---
        for start in tqdm(range(0, len(valid_anchor_indices), args.batch_size), desc=f"Mining {data_split} batches"):
            end = min(start + args.batch_size, len(valid_anchor_indices))
            batch_local_indices = valid_anchor_indices[start:end]
            if len(batch_local_indices) == 0: continue
            batch_embeddings = split_embeddings[batch_local_indices]
            scores, indices = gpu_index.search(batch_embeddings, args.num_candidates_faiss)

            for i, anchor_local_idx in enumerate(batch_local_indices):
                anchor_id = split_all_ids[anchor_local_idx]
                anchor_meta = {'binary_name': split_all_binary_names[anchor_local_idx], 'function_name': split_all_function_names[anchor_local_idx]}

                # --- NEW HYBRID SAMPLING LOGIC ---
                # 1. Classify all candidates into buckets
                hard_pos_candidates_gt = []
                hard_neg_candidates = []
                easy_neg_candidates = []
                
                for cand_local_idx, score in zip(indices[i], scores[i]):
                    if cand_local_idx == -1 or cand_local_idx == anchor_local_idx: continue
                    cand_id = split_all_ids[cand_local_idx]
                    cand_meta = {'binary_name': split_all_binary_names[cand_local_idx], 'function_name': split_all_function_names[cand_local_idx]}
                    
                    is_positive = (anchor_meta['function_name'] == cand_meta['function_name'] and anchor_meta['binary_name'] == cand_meta['binary_name'])

                    if is_positive:
                        if score >= args.hard_positive_threshold:
                            hard_pos_candidates_gt.append(cand_id)
                    else: # Is a negative
                        if args.hard_negative_min_score <= score < args.hard_negative_max_score:
                            hard_neg_candidates.append(cand_id)
                        elif score < args.hard_negative_min_score:
                            easy_neg_candidates.append(cand_id)

                # 2. Select exactly ONE positive
                chosen_positive_id = None
                if hard_pos_candidates_gt:
                    chosen_positive_id = random.choice(hard_pos_candidates_gt)
                else:
                    all_gt_for_anchor = gt_lookup_split[anchor_meta['binary_name']][anchor_meta['function_name']]
                    valid_positives = [pid for pid in all_gt_for_anchor if pid != anchor_id]
                    chosen_positive_id = random.choice(valid_positives) # Guaranteed to not be empty
                
                final_positives = [chosen_positive_id]
                
                # 3. Sample negatives based on quotas
                final_negatives = []
                
                # Sample hard negatives
                num_hard_to_sample = min(len(hard_neg_candidates), args.num_hard_negatives)
                if num_hard_to_sample > 0:
                    final_negatives.extend(random.sample(hard_neg_candidates, num_hard_to_sample))
                    
                # Sample "easy" negatives from FAISS results first
                num_random_to_sample = args.num_random_negatives
                num_easy_from_faiss = min(len(easy_neg_candidates), num_random_to_sample)
                if num_easy_from_faiss > 0:
                    final_negatives.extend(random.sample(easy_neg_candidates, num_easy_from_faiss))

                # Backfill with purely random negatives if quotas not met
                remaining_needed = (args.top_k - 1) - len(final_negatives)
                if remaining_needed > 0:
                    all_gt_for_anchor = gt_lookup_split[anchor_meta['binary_name']][anchor_meta['function_name']]
                    forbidden_ids = set(all_gt_for_anchor)
                    forbidden_ids.update(final_negatives)
                    
                    while len(final_negatives) < (args.top_k - 1):
                        rand_idx = random.randint(0, split_num_rows - 1)
                        rand_id = split_all_ids[rand_idx]
                        if rand_id not in forbidden_ids:
                            final_negatives.append(rand_id)
                            forbidden_ids.add(rand_id)
                
                # --- Calculate scores and append ---
                anchor_embedding = split_embeddings[anchor_local_idx]
                pos_indices = [split_id_to_index[pid] for pid in final_positives]
                pos_embeddings = split_embeddings[pos_indices]
                pos_scores = (anchor_embedding @ pos_embeddings.T).tolist()
                neg_indices = [split_id_to_index[nid] for nid in final_negatives]
                neg_embeddings = split_embeddings[neg_indices]
                neg_scores = (anchor_embedding @ neg_embeddings.T).tolist()
                
                final_anchor_ids.append(anchor_id)
                final_positive_ids.append(final_positives)
                final_negative_ids.append(final_negatives)
                final_splits.append(data_split)
                final_positive_scores.append(pos_scores)
                final_negative_scores.append(neg_scores)

    # --- Step 3: Build and save the COMBINED final dataset ---
    if not final_anchor_ids:
        print("No valid pairs found across all splits.")
    else:
        final_ds = Dataset.from_dict({
            "unique_id": final_anchor_ids,
            "positive_ids": final_positive_ids,
            "negative_ids": final_negative_ids,
            "positive_scores": final_positive_scores,
            "negative_scores": final_negative_scores,
            "split": final_splits
        })
        output_path = os.path.join(args.output_dir, split_path)
        print(f"Saving combined dataset to {output_path}")
        final_ds.save_to_disk(output_path)

    print("\nAll splits processed successfully!")