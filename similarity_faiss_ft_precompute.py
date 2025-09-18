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
    parser = argparse.ArgumentParser(description="FAISS-First Hybrid dataset generation for fine-tuning.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_faiss_ft")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_candidates_faiss", type=int, default=64, help="Number of candidates to initially retrieve from FAISS. Should be > top_k.")
    parser.add_argument("--top_k", type=int, default=10, help="The final total number of targets for each anchor.")
    parser.add_argument("--split", type=str, default='project')
    args = parser.parse_args()
    split_path = f'cross_{args.split}_split'
    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    assert args.num_candidates_faiss > args.top_k, "--num_candidates_faiss must be greater than --top_k"

    # --- Setup: Load full dataset once to create splits ---
    print("Loading full dataset to prepare splits...")
    original_dataset = load_from_disk(args.data_dir)
    split_file_path = os.path.join(args.splits_dir, f"{split_path}.json")
    with open(split_file_path, 'r') as f:
        split_ids_by_group = json.load(f)

    # --- Master Loop: Process each split IN ISOLATION ---
    final_anchor_ids, final_positive_ids, final_negative_ids, final_splits = [], [], [], []

    for data_split in ["train", "val"]:
        print(f"\n{'='*20} PROCESSING SPLIT: {data_split.upper()} {'='*20}")
        if data_split not in split_ids_by_group: continue

        # --- Step 1: Create the isolated universe for this split ---
        split_ids_set = set(split_ids_by_group[data_split])
        split_dataset = original_dataset.filter(lambda x: x['unique_id'] in split_ids_set, num_proc=32)
        if len(split_dataset) == 0: continue
        
        # --- PERFORMANCE OPTIMIZATION: Bulk extract all data into fast structures ---
        split_dataset.set_format("numpy", columns=['unique_id', 'binary_name', 'function_name', 'clap_embedding'])
        split_data_np = split_dataset[:]
        split_embeddings = split_data_np['clap_embedding'].astype('float32')
        split_all_ids = split_data_np['unique_id'].tolist()
        # Create parallel lists for metadata - this is the key fix
        split_all_binary_names = split_data_np['binary_name'].tolist()
        split_all_function_names = split_data_np['function_name'].tolist()
        # --------------------------------------------------------------------------
        
        split_num_rows, dim = split_embeddings.shape
        split_id_to_index = {uid: i for i, uid in enumerate(split_all_ids)}
        
        gt_lookup_split = {}
        # Use the fast lists we just created
        for i in tqdm(range(split_num_rows), desc=f"Building {data_split} GT lookup"):
            uid_int = int(split_all_ids[i])
            b_name_str = str(split_all_binary_names[i])
            f_name_str = str(split_all_function_names[i])
            if b_name_str not in gt_lookup_split: gt_lookup_split[b_name_str] = {}
            if f_name_str not in gt_lookup_split[b_name_str]: gt_lookup_split[b_name_str][f_name_str] = []
            gt_lookup_split[b_name_str][f_name_str].append(uid_int)

        nlist = max(1, min(4096, split_num_rows // 100))
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        cpu_index.train(split_embeddings[np.random.choice(split_num_rows, size=min(split_num_rows, 256*1024), replace=False)])
        cpu_index.add(split_embeddings)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.nprobe = 32

        # --- Step 2: Process all anchors in the split ---
        anchor_indices_in_split = np.arange(split_num_rows)
        for start in tqdm(range(0, len(anchor_indices_in_split), args.batch_size), desc=f"Mining {data_split} batches"):
            end = min(start + args.batch_size, len(anchor_indices_in_split))
            batch_local_indices = anchor_indices_in_split[start:end]
            batch_embeddings = split_embeddings[batch_local_indices]
            
            scores, indices = gpu_index.search(batch_embeddings, args.num_candidates_faiss)

            for i, anchor_local_idx in enumerate(batch_local_indices):
                anchor_id = split_all_ids[anchor_local_idx]
                
                # --- PERFORMANCE OPTIMIZATION: Use fast list lookups ---
                anchor_meta = {'binary_name': split_all_binary_names[anchor_local_idx], 
                               'function_name': split_all_function_names[anchor_local_idx]}

                positives_found_by_faiss, negatives_found_by_faiss = [], []
                for cand_local_idx, score in zip(indices[i], scores[i]):
                    if cand_local_idx == -1 or cand_local_idx == anchor_local_idx: continue
                    cand_id = split_all_ids[cand_local_idx]
                    
                    # Use fast list lookups here as well
                    cand_meta = {'binary_name': split_all_binary_names[cand_local_idx], 
                                 'function_name': split_all_function_names[cand_local_idx]}
                # ----------------------------------------------------------

                    if (anchor_meta['function_name'] == cand_meta['function_name'] and anchor_meta['binary_name'] == cand_meta['binary_name']):
                        positives_found_by_faiss.append(cand_id)
                    else:
                        negatives_found_by_faiss.append(cand_id)
                
                # ... (The rest of your excellent sampling logic remains unchanged) ...
                final_positives = positives_found_by_faiss
                if not final_positives:
                    all_gt_for_anchor = gt_lookup_split[anchor_meta['binary_name']][anchor_meta['function_name']]
                    valid_positives = [pid for pid in all_gt_for_anchor if pid != anchor_id]
                    if valid_positives:
                        final_positives.append(random.choice(valid_positives))

                if not final_positives: continue

                num_negatives_to_take = args.top_k - len(final_positives)
                final_negatives = negatives_found_by_faiss[:num_negatives_to_take]

                remaining_needed = num_negatives_to_take - len(final_negatives)
                if remaining_needed > 0:
                    all_gt_for_anchor = gt_lookup_split[anchor_meta['binary_name']][anchor_meta['function_name']]
                    forbidden_ids = set(all_gt_for_anchor)
                    forbidden_ids.update(final_negatives)
                    
                    while len(final_negatives) < num_negatives_to_take:
                        rand_idx = random.randint(0, split_num_rows - 1)
                        rand_id = split_all_ids[rand_idx]
                        if rand_id not in forbidden_ids:
                            final_negatives.append(rand_id)
                            forbidden_ids.add(rand_id)
                
                final_anchor_ids.append(anchor_id)
                final_positive_ids.append(final_positives)
                final_negative_ids.append(final_negatives)
                final_splits.append(data_split)

    # --- Step 3: Build and save the COMBINED final dataset ---
    if not final_anchor_ids:
        print("No valid pairs found across all splits.")
    else:
        final_ds = Dataset.from_dict({
            "unique_id": final_anchor_ids,
            "positive_ids": final_positive_ids,
            "negative_ids": final_negative_ids,
            "split": final_splits
        })
        print(f"\nFinal dataset stats:\n{final_ds}")
        output_path = os.path.join(args.output_dir, split_path)
        print(f"Saving combined dataset to {output_path}")
        final_ds.save_to_disk(output_path)

    print("\nAll splits processed successfully!")