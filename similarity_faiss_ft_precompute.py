import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import numpy as np
import argparse
from datasets import load_from_disk, Dataset
import gc
from tqdm import tqdm
import faiss
import random

# This is the "ultimate" script. It combines the "Guaranteed Positive Hybrid" strategy
# with the strict definition of hard negatives based on teacher scores.
# It produces the highest quality data for both fine-tuning and distillation.

if __name__ == "__main__":
    # --- Arguments ---
    parser = argparse.ArgumentParser(description="Generate the highest-quality hybrid dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the original dataset with embeddings and metadata.")
    parser.add_argument("--splits_dir", type=str, required=True, help="Directory with JSON files defining train/val splits.")
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_faiss_ft")
    
    # --- Performance Tuning Arguments ---
    parser.add_argument("--batch_size", type=int, default=8192, help="Anchors to process per FAISS search batch.")
    parser.add_argument("--num_candidates_faiss", type=int, default=512, help="Nearest neighbors to retrieve from FAISS.")
    
    # --- NEW & IMPROVED: Arguments for controlling negative composition ---
    parser.add_argument("--top_k", type=int, default=10, help="Total number of targets (1 positive + k-1 negatives).")
    parser.add_argument("--num_hard_negatives", type=int, default=7, help="Number of STRICT hard negatives to prioritize.")
    
    # --- NEW & IMPROVED: Re-introduced score thresholds ---
    parser.add_argument("--hard_negative_max_score", type=float, default=0.7)
    parser.add_argument("--hard_negative_min_score", type=float, default=0.2)

    parser.add_argument("--debug_subset_size", type=int, default=None, 
                    help="Run on a small subset of N anchors for debugging. If not set, runs on the full dataset.")
    
    random.seed(42)
    np.random.seed(42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    num_negatives_to_sample = args.top_k - 1
    assert args.num_hard_negatives <= num_negatives_to_sample, "num_hard_negatives cannot be greater than total negatives."

    # --- 1. Load data and build lookups (same as before) ---
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

    metadata_lookup = {}
    gt_lookup = {}
    for uid, b_name, f_name in tqdm(zip(all_data_np['unique_id'], all_data_np['binary_name'], all_data_np['function_name']), total=len(all_data_np['unique_id']), desc="Building metadata & GT lookups"):
        uid_int, b_name_str, f_name_str = int(uid), str(b_name), str(f_name)
        metadata_lookup[uid_int] = {'binary_name': b_name_str, 'function_name': f_name_str}
        if b_name_str not in gt_lookup: gt_lookup[b_name_str] = {}
        if f_name_str not in gt_lookup[b_name_str]: gt_lookup[b_name_str][f_name_str] = []
        gt_lookup[b_name_str][f_name_str].append(uid_int)
    
    del all_data_np
    print("Lookups built.")

    # --- 2. Build FAISS Index (same as before) ---
    print("Building FAISS Index...")
    # ... (FAISS code is identical) ...
    nlist = 4096 
    quantizer = faiss.IndexFlatIP(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    train_sample_size = min(num_rows, 256 * 1024)
    cpu_index.train(embeddings[np.random.choice(num_rows, size=train_sample_size, replace=False)])
    cpu_index.add(embeddings)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.nprobe = 32
    print("Index is on GPU.")

    # --- 3. Process each split JSON file ---
    for split_file in os.listdir(args.splits_dir):
        # ... (file handling is identical) ...
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
            print(f"--- Starting ultimate hybrid mining for '{data_split}' set ---")
            split_ids = split_ids_by_group[data_split]
            if args.debug_subset_size is not None:
                print(f"!!! RUNNING IN DEBUG MODE ON A SUBSET OF {args.debug_subset_size} ANCHORS !!!")
                split_ids = split_ids[:args.debug_subset_size]
            anchor_indices = np.array([id_to_index.get(uid) for uid in split_ids if id_to_index.get(uid) is not None])

            for start in tqdm(range(0, len(anchor_indices), args.batch_size), desc=f"Processing {data_split} batches"):
                end = min(start + args.batch_size, len(anchor_indices))
                batch_anchor_indices = anchor_indices[start:end]
                
                if len(batch_anchor_indices) == 0: continue
                batch_anchor_embeddings = embeddings[batch_anchor_indices]

                scores, indices = gpu_index.search(batch_anchor_embeddings, args.num_candidates_faiss)
                
                for i in range(len(batch_anchor_indices)):
                    anchor_idx = batch_anchor_indices[i]
                    anchor_id = all_ids[anchor_idx]
                    anchor_meta = metadata_lookup[anchor_id]
                    
                    faiss_candidate_indices = indices[i]
                    faiss_candidate_scores = scores[i]
                    
                    # ### NEW & IMPROVED: Partition FAISS results into 3 groups ###
                    gt_positives_in_faiss = []
                    strict_hard_negs_in_faiss = []
                    other_negs_in_faiss = []

                    for cand_idx, score in zip(faiss_candidate_indices, faiss_candidate_scores):
                        if cand_idx == -1 or cand_idx == anchor_idx: continue
                        target_meta = metadata_lookup.get(cand_idx)
                        if not target_meta: continue

                        # Check if it's a GT positive
                        if (anchor_meta['function_name'] == target_meta['function_name'] and
                            anchor_meta['binary_name'] == target_meta['binary_name']):
                            gt_positives_in_faiss.append(cand_idx)
                        else:
                            # If it's a GT negative, check if it's "strictly hard"
                            if args.hard_negative_min_score <= score < args.hard_negative_max_score:
                                strict_hard_negs_in_faiss.append(cand_idx)
                            else:
                                other_negs_in_faiss.append(cand_idx)
                    
                    # --- HYBRID LOGIC TO SELECT A GUARANTEED POSITIVE ---
                    guaranteed_positive_idx = None
                    if len(gt_positives_in_faiss) > 0:
                        guaranteed_positive_idx = random.choice(gt_positives_in_faiss)
                    else:
                        all_gt_positives = gt_lookup[anchor_meta['binary_name']][anchor_meta['function_name']]
                        for pid in all_gt_positives:
                            if pid != anchor_id:
                                guaranteed_positive_idx = id_to_index[pid]
                                break
                    
                    if guaranteed_positive_idx is not None:
                        # --- PRIORITY SAMPLING FOR NEGATIVES ---
                        sampled_negatives = []
                        
                        # 1. Prioritize strict hard negatives
                        num_hard_to_sample = min(len(strict_hard_negs_in_faiss), args.num_hard_negatives)
                        if num_hard_to_sample > 0:
                            sampled_negatives.extend(random.sample(strict_hard_negs_in_faiss, num_hard_to_sample))
                        
                        # 2. Fill with other FAISS negatives if needed
                        remaining_needed = num_negatives_to_sample - len(sampled_negatives)
                        if remaining_needed > 0 and other_negs_in_faiss:
                            num_other_to_sample = min(len(other_negs_in_faiss), remaining_needed)
                            sampled_negatives.extend(random.sample(other_negs_in_faiss, num_other_to_sample))
                        
                        # 3. Fill the rest with purely random negatives
                        remaining_needed = num_negatives_to_sample - len(sampled_negatives)
                        if remaining_needed > 0:
                            forbidden = set(sampled_negatives)
                            forbidden.add(anchor_idx)
                            forbidden.add(guaranteed_positive_idx)
                            
                            while len(sampled_negatives) < num_negatives_to_sample:
                                rand_idx = random.randint(0, num_rows - 1)
                                if rand_idx not in forbidden:
                                    sampled_negatives.append(rand_idx)
                                    forbidden.add(rand_idx)
                        
                        # --- Construct final target list and save ---
                        final_target_indices = [guaranteed_positive_idx] + sampled_negatives
                        target_scores = (embeddings[anchor_idx] @ embeddings[final_target_indices].T).tolist()
                        target_ids = [all_ids[idx] for idx in final_target_indices]
                        
                        final_anchor_ids.append(anchor_id)
                        final_target_ids.append(target_ids)
                        final_cosine_scores.append(target_scores)
                        final_splits.append(data_split)

            gc.collect()
        
        # --- 5. Build and save the final dataset (same as before) ---
        if not final_anchor_ids: continue
        final_ds = Dataset.from_dict({
            "unique_id": final_anchor_ids,
            "target_ids": final_target_ids,
            "cosine_scores": final_cosine_scores,
            "split": final_splits
        })
        output_path = os.path.join(args.output_dir, split_name)
        print(f"Saving ultimate hybrid dataset to {output_path}")
        final_ds.save_to_disk(output_path)
        del final_ds
        gc.collect()

    print("\nAll splits processed successfully!")