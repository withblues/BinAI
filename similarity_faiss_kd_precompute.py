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

if __name__ == "__main__":
    # --- Arguments (Consider tuning batch_size and num_candidates_faiss) ---
    parser = argparse.ArgumentParser(description="Generate Hard Negative Pools with FAISS for Embeddings.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_hard_mined_pool_fast")
    parser.add_argument("--batch_size", type=int, default=4096, help="Anchors to process per batch. Increase for better GPU utilization.") # TUNABLE
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_candidates_faiss", type=int, default=512, help="Nearest neighbors to retrieve. Can be reduced for speed.") # TUNABLE
    # ... rest of the arguments are the same ...
    parser.add_argument("--num_hard_positives", type=int, default=1)
    parser.add_argument("--num_hard_negatives", type=int, default=5)
    parser.add_argument("--num_random_negatives", type=int, default=4)
    parser.add_argument("--hard_positive_threshold", type=float, default=0.7)
    parser.add_argument("--hard_negative_max_score", type=float, default=0.7)
    parser.add_argument("--hard_negative_min_score", type=float, default=0.2)
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    
    # ... Assertions ...
    assert (args.num_hard_positives + args.num_hard_negatives + args.num_random_negatives == args.top_k)

    # --- 1. Load full dataset ---
    print("Loading full dataset...")
    dataset = load_from_disk(args.data_dir)
    if "clap_embedding" in dataset.column_names:
        dataset = dataset.rename_column("clap_embedding", "embedding")
    all_ids = list(dataset["unique_id"])
    dataset.set_format("numpy", columns=["embedding"])
    embeddings = dataset[:]["embedding"].astype('float32')
    num_rows, dim = embeddings.shape
    print(f"Full dataset loaded. Embeddings shape: {embeddings.shape}")
    id_to_index = {uid: i for i, uid in enumerate(all_ids)}

    # --- 2. Build an OPTIMIZED Approximate Faiss Index ---
    # Heuristic for nlist: k * sqrt(N), where k is a small integer. 4096 is a good start for 4M vectors.
    nlist = 4096 
    quantizer = faiss.IndexFlatIP(dim) # The quantizer is the 'map' of the cells
    
    # --- CHANGE 1: Switched to IndexIVFFlat ---
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print("Training the FAISS index...")
    # The index needs to be trained on a sample of the data to learn the clusters.
    # For large datasets, training on a subset is common and efficient.
    # Let's train on up to 256,000 vectors.
    train_sample_size = min(num_rows, 256 * 1024)
    random_indices = np.random.choice(num_rows, size=train_sample_size, replace=False)
    cpu_index.train(embeddings[random_indices])
    print("Training complete.")
    
    print("Adding vectors to the index...")
    cpu_index.add(embeddings)
    print(f"Index built with {cpu_index.ntotal} vectors.")

    print("Moving index to GPU...")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # --- IMPORTANT: Set nprobe ---
    # This controls the speed/accuracy trade-off. Higher is more accurate but slower.
    # Start with a low value and increase if recall is too low.
    gpu_index.nprobe = 32 # TUNABLE
    print(f"Index is on GPU. nprobe set to {gpu_index.nprobe}")


    # --- 3. Process each split JSON file ---
    for split_file in os.listdir(args.splits_dir):
        # ... (file checking logic is the same) ...
        if not split_file.endswith(".json"): continue
        split_name = os.path.splitext(split_file)[0]
        print(f"\nProcessing split file: {split_name}")
        json_path = os.path.join(args.splits_dir, split_file)
        if os.path.getsize(json_path) == 0: continue
        with open(json_path, 'r') as f:
            split_ids_by_group = json.load(f)

        final_unique_ids, final_split_column, final_target_ids, final_cosine_scores = [], [], [], []

        for data_split in ["train", "val"]:
            if data_split not in split_ids_by_group: continue
            print(f"--- Starting hard mining for '{data_split}' set ---")
            split_ids = split_ids_by_group[data_split]
            anchor_indices = np.array([id_to_index[uid] for uid in split_ids])

            for start in tqdm(range(0, len(anchor_indices), args.batch_size), desc=f"Processing {data_split} batches"):
                end = min(start + args.batch_size, len(anchor_indices))
                batch_anchor_indices = anchor_indices[start:end]
                batch_anchor_embeddings = embeddings[batch_anchor_indices]

                scores, indices = gpu_index.search(batch_anchor_embeddings, args.num_candidates_faiss + 1)
                
                # --- CHANGE 2: Vectorized post-processing ---
                # Create boolean masks for the entire batch at once
                self_mask = indices == batch_anchor_indices[:, np.newaxis]
                
                pos_mask = (scores >= args.hard_positive_threshold) & ~self_mask
                hard_neg_mask = (scores >= args.hard_negative_min_score) & (scores < args.hard_negative_max_score) & ~self_mask
                
                # Now loop through and sample from the pre-filtered candidates
                for i in range(len(batch_anchor_indices)):
                    anchor_global_idx = batch_anchor_indices[i]
                    anchor_id = all_ids[anchor_global_idx]
                    
                    # Get candidates for this specific anchor using the masks
                    pos_candidates = indices[i][pos_mask[i]]
                    hard_neg_candidates = indices[i][hard_neg_mask[i]]
                    
                    selected_targets = []

                    # Sample hard positives
                    if len(pos_candidates) > 0:
                        num_to_add = min(len(pos_candidates), args.num_hard_positives)
                        selected_indices = np.random.choice(pos_candidates, num_to_add, replace=False)
                        selected_targets.extend(selected_indices.tolist())

                    # Sample hard negatives
                    if len(hard_neg_candidates) > 0:
                        num_to_add = min(len(hard_neg_candidates), args.num_hard_negatives)
                        selected_indices = np.random.choice(hard_neg_candidates, num_to_add, replace=False)
                        selected_targets.extend(selected_indices.tolist())

                    # Fill with random negatives (this part is already reasonably fast)
                    num_random_to_add = args.top_k - len(selected_targets)
                    if num_random_to_add > 0:
                        current_targets = set(selected_targets)
                        current_targets.add(anchor_global_idx)
                        
                        # Generate a pool of random candidates and filter out existing ones
                        random_pool = np.random.randint(0, num_rows, size=num_random_to_add * 5)
                        valid_randoms = [idx for idx in random_pool if idx not in current_targets]
                        selected_targets.extend(valid_randoms[:num_random_to_add])
                    
                    # Fallback
                    while len(selected_targets) < args.top_k:
                        rand_idx = random.randint(0, num_rows - 1)
                        if rand_idx != anchor_global_idx and rand_idx not in selected_targets:
                            selected_targets.append(rand_idx)
                            
                    selected_targets = selected_targets[:args.top_k]

                    # Finalize and shuffle
                    target_scores = (embeddings[anchor_global_idx] @ embeddings[selected_targets].T).tolist()
                    target_ids = [all_ids[idx] for idx in selected_targets]
                    
                    combined = list(zip(target_ids, target_scores))
                    random.shuffle(combined)
                    shuffled_ids, shuffled_scores = zip(*combined) if combined else ([], [])

                    final_unique_ids.append(anchor_id)
                    final_split_column.append(data_split)
                    final_target_ids.append(list(shuffled_ids))
                    final_cosine_scores.append(list(shuffled_scores))

            gc.collect()

        # --- 5. Build and save the final dataset ---
        if not final_unique_ids: continue
        final_ds = Dataset.from_dict({ "unique_id": final_unique_ids, "split": final_split_column, "target_ids": final_target_ids, "cosine_scores": final_cosine_scores })
        output_path = os.path.join(args.output_dir, split_name)
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving combined hard-mined dataset to {output_path}")
        final_ds.save_to_disk(output_path)
        del final_ds, final_unique_ids, final_split_column, final_target_ids, final_cosine_scores
        gc.collect()

    print("\nAll splits processed successfully!")