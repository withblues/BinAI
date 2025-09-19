import os
import json
import numpy as np
import argparse
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import random

if __name__ == "__main__":
    # ... (Arguments are the same) ...
    parser = argparse.ArgumentParser(description="Generate a random baseline dataset for Fine-Tuning with scores.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_random_ft")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default='project')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    split_path = f'cross_{args.split}_split'
    
    print("Loading full dataset and split definitions...")
    original_dataset = load_from_disk(args.data_dir)
    split_file_path = os.path.join(args.splits_dir, f"{split_path}.json")
    with open(split_file_path, 'r') as f:
        split_ids_by_group = json.load(f)
        
    final_anchor_ids, final_positive_ids, final_negative_ids = [], [], []
    final_positive_scores, final_negative_scores, final_splits = [], [], []

    for data_split in ["train", "val"]:
        print(f"\n{'='*20} PROCESSING SPLIT: {data_split.upper()} {'='*20}")
        if data_split not in split_ids_by_group: continue

        split_ids_set = set(split_ids_by_group[data_split])
        split_dataset = original_dataset.filter(lambda x: x['unique_id'] in split_ids_set, num_proc=32)
        if len(split_dataset) == 0: continue
        
        # --- THE FIX: Load all necessary columns, including embeddings ---
        split_dataset.set_format("numpy", columns=['unique_id', 'binary_name', 'function_name', 'clap_embedding'])
        split_data_np = split_dataset[:]
        
        # Now create all the necessary fast-access lists and arrays
        split_embeddings = split_data_np['clap_embedding'].astype('float32')
        split_all_ids = split_data_np['unique_id'].tolist()
        split_all_binary_names = split_data_np['binary_name'].tolist()
        split_all_function_names = split_data_np['function_name'].tolist()
        # -----------------------------------------------------------------

        split_num_rows = len(split_all_ids)
        split_id_to_index = {uid: i for i, uid in enumerate(split_all_ids)}
        split_gt_lookup = {}
        for i in tqdm(range(split_num_rows), desc=f"Building {data_split} GT lookup"):
            uid_int, b_name_str, f_name_str = int(split_all_ids[i]), str(split_all_binary_names[i]), str(split_all_function_names[i])
            if b_name_str not in split_gt_lookup: split_gt_lookup[b_name_str] = {}
            if f_name_str not in split_gt_lookup[b_name_str]: split_gt_lookup[b_name_str][f_name_str] = []
            split_gt_lookup[b_name_str][f_name_str].append(uid_int)

        anchor_ids_for_split = split_ids_by_group[data_split]
        possible_negative_indices = list(range(split_num_rows))

        for anchor_id in tqdm(anchor_ids_for_split, desc=f"Mining {data_split} pairs"):
            if anchor_id not in split_id_to_index: continue
            
            anchor_idx_in_split = split_id_to_index[anchor_id]
            # Use fast list lookups for metadata
            anchor_meta = {'binary_name': split_all_binary_names[anchor_idx_in_split], 
                           'function_name': split_all_function_names[anchor_idx_in_split]}

            all_gt_positives = split_gt_lookup[str(anchor_meta['binary_name'])][str(anchor_meta['function_name'])]
            valid_gt_positives = [pid for pid in all_gt_positives if pid != anchor_id]
            if not valid_gt_positives: continue
            
            chosen_positive_id = random.choice(valid_gt_positives)
            final_positives = [chosen_positive_id]
            
            num_negatives_to_sample = args.top_k - 1
            
            forbidden_indices = {split_id_to_index[pid] for pid in all_gt_positives}
                
            sampled_negatives_indices = []
            while len(sampled_negatives_indices) < num_negatives_to_sample:
                rand_idx = random.choice(possible_negative_indices)
                if rand_idx not in forbidden_indices:
                    sampled_negatives_indices.append(rand_idx)
                    forbidden_indices.add(rand_idx)
            
            final_negatives = [split_all_ids[idx] for idx in sampled_negatives_indices]

            # This part now works correctly
            anchor_embedding = split_embeddings[anchor_idx_in_split]
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

    if not final_anchor_ids:
        print("No valid pairs found.")
    else:
        final_ds = Dataset.from_dict({
            "unique_id": final_anchor_ids,
            "positive_ids": final_positive_ids,
            "negative_ids": final_negative_ids,
            "positive_scores": final_positive_scores,
            "negative_scores": final_negative_scores,
            "split": final_splits
        })
        print(f"\nFinal dataset stats:\n{final_ds}")
        output_path = os.path.join(args.output_dir, split_path)
        print(f"Saving combined dataset to {output_path}")
        final_ds.save_to_disk(output_path)

    print("\nAll splits processed successfully!")