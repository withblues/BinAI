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
    # ... (Arguments are the same) ...
    parser = argparse.ArgumentParser(description="FAISS-based dataset generation for fine-tuning (1 positive vs k-1 negatives).")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_faiss_ft_1_vs_k")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_candidates_faiss", type=int, default=512, help="Number of candidates to initially retrieve from FAISS. A larger value finds better hard negatives.")
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
        
        # ... (Bulk data extraction is the same and correct) ...
        split_dataset.set_format("numpy", columns=['unique_id', 'binary_name', 'function_name', 'clap_embedding'])
        split_data_np = split_dataset[:]
        split_embeddings = split_data_np['clap_embedding'].astype('float32')
        split_all_ids = split_data_np['unique_id'].tolist()
        split_all_binary_names = split_data_np['binary_name'].tolist()
        split_all_function_names = split_data_np['function_name'].tolist()
        split_num_rows, dim = split_embeddings.shape
        split_id_to_index = {uid: i for i, uid in enumerate(split_all_ids)}
        gt_lookup_split = {}
        for i in tqdm(range(split_num_rows), desc=f"Building {data_split} GT lookup"):
            uid_int, b_name_str, f_name_str = int(split_all_ids[i]), str(split_all_binary_names[i]), str(split_all_function_names[i])
            if b_name_str not in gt_lookup_split: gt_lookup_split[b_name_str] = {}
            if f_name_str not in gt_lookup_split[b_name_str]: gt_lookup_split[b_name_str][f_name_str] = []
            gt_lookup_split[b_name_str][f_name_str].append(uid_int)

        # --- Create a list of ANCHORS THAT ARE ACTUALLY VALID ---
        # This is a pre-filtering step to improve efficiency
        valid_anchor_indices = []
        for idx in range(split_num_rows):
             meta = {'binary_name': split_all_binary_names[idx], 'function_name': split_all_function_names[idx]}
             all_gt = gt_lookup_split[meta['binary_name']][meta['function_name']]
             if len(all_gt) > 1: # Check if there is at least one OTHER positive
                 valid_anchor_indices.append(idx)
        
        valid_anchor_indices = np.array(valid_anchor_indices)
        print(f"Found {len(valid_anchor_indices)} valid anchors with at least one positive pair.")

        # --- Build FAISS index ---
        # ... (FAISS building is the same and correct) ...
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

                positives_found_by_faiss, negatives_found_by_faiss = [], []
                for cand_local_idx, score in zip(indices[i], scores[i]):
                    if cand_local_idx == -1 or cand_local_idx == anchor_local_idx: continue
                    cand_id = split_all_ids[cand_local_idx]
                    cand_meta = {'binary_name': split_all_binary_names[cand_local_idx], 'function_name': split_all_function_names[cand_local_idx]}
                    if (anchor_meta['function_name'] == cand_meta['function_name'] and anchor_meta['binary_name'] == cand_meta['binary_name']):
                        positives_found_by_faiss.append(cand_id)
                    else:
                        negatives_found_by_faiss.append(cand_id)
                
                # --- LOGIC REMAINS THE SAME, BUT THE PRE-CONDITION IS ALREADY MET ---
                chosen_positive_id = None
                if positives_found_by_faiss:
                    chosen_positive_id = random.choice(positives_found_by_faiss)
                else:
                    all_gt_for_anchor = gt_lookup_split[anchor_meta['binary_name']][anchor_meta['function_name']]
                    valid_positives = [pid for pid in all_gt_for_anchor if pid != anchor_id]
                    # This list is now guaranteed to not be empty because of our pre-filtering
                    chosen_positive_id = random.choice(valid_positives)

                final_positives = [chosen_positive_id]
                num_negatives_to_take = args.top_k - 1
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
        # ... (Saving logic is the same) ...
        final_ds = Dataset.from_dict({
            "unique_id": final_anchor_ids,
            "positive_ids": final_positive_ids,
            "negative_ids": final_negative_ids,
            "split": final_splits
        })
        output_path = os.path.join(args.output_dir, split_path)
        final_ds.save_to_disk(output_path)

    print("\nAll splits processed successfully!")