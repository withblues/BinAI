import os
import json
import numpy as np
import argparse
from datasets import load_from_disk, Dataset
import gc
from tqdm import tqdm

def generate_random_indices(num_rows, top_k, seed=42):
    np.random.seed(seed)
    all_target_indices = np.empty((num_rows, top_k), dtype=np.int64)
    for i in tqdm(range(num_rows), desc="Generating random indices"):
        sampled = set()
        while len(sampled) < top_k:
            idx = np.random.randint(num_rows)
            if idx != i:
                sampled.add(idx)
        all_target_indices[i] = np.array(list(sampled))
    return all_target_indices

def process_batch(batch_embeddings, batch_indices, split_ids_list):
    # batch_embeddings: (batch_size, dim)
    # batch_indices: (batch_size, top_k)
    target_embeddings = embeddings[batch_indices]  # shape: (batch_size, top_k, dim)
    cosine_scores = np.einsum('bd,bkd->bk', batch_embeddings, target_embeddings)

    # map indices to unique_ids
    target_ids = np.array([[split_ids_list[idx] for idx in row] for row in batch_indices], dtype=object)
    return target_ids.tolist(), cosine_scores.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random negative pools for embeddings.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_pools")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    # --- Load full dataset once ---
    print(f"Loading full dataset from {args.data_dir}")
    dataset = load_from_disk(args.data_dir)
    if "clap_embedding" in dataset.column_names:
        dataset = dataset.rename_column("clap_embedding", "embedding")

    # Materialize unique IDs first
    all_ids = list(dataset["unique_id"])

    # Convert embeddings column to NumPy
    dataset.set_format("numpy", columns=["embedding"])
    embeddings = dataset[:]["embedding"]
    num_rows, dim = embeddings.shape
    print(f"Embeddings shape: {embeddings.shape}")

    # Map unique_id -> row index for fast lookup
    id_to_index = {uid: i for i, uid in enumerate(all_ids)}

    # --- Process each split JSON ---
    for split_file in os.listdir(args.splits_dir):
        if not split_file.endswith(".json"):
            continue
        split_name = os.path.splitext(split_file)[0]
        print(f"\nProcessing split file: {split_name}")

        json_path = os.path.join(args.splits_dir, split_file)
        if os.path.getsize(json_path) == 0:
            print(f"Skipping empty file: {split_file}")
            continue

        with open(json_path, 'r') as f:
            split_ids_by_group = json.load(f)

        # --- Combine train + val ---
        combined_split_ids = []
        combined_split_embeddings = []
        combined_unique_ids = []
        split_column = []

        for data_split in ["train", "val"]:
            if data_split not in split_ids_by_group:
                continue
            split_ids = split_ids_by_group[data_split]
            split_indices = [id_to_index[uid] for uid in split_ids]
            split_embeddings = embeddings[split_indices]
            split_unique_ids = [all_ids[i] for i in split_indices]

            combined_split_ids.extend(split_ids)
            combined_split_embeddings.append(split_embeddings)
            combined_unique_ids.extend(split_unique_ids)
            split_column.extend([data_split] * len(split_ids))

        combined_split_embeddings = np.vstack(combined_split_embeddings)
        num_combined_rows = combined_split_embeddings.shape[0]
        print(f"Combined train+val rows: {num_combined_rows}")

        # --- Generate random target indices per split ---
        all_target_indices = np.empty((num_combined_rows, args.top_k), dtype=np.int64)
        start_idx = 0
        for data_split in ["train", "val"]:
            if data_split not in split_ids_by_group:
                continue
            split_ids = split_ids_by_group[data_split]
            split_size = len(split_ids)
            all_target_indices[start_idx:start_idx+split_size] = generate_random_indices(split_size, args.top_k)
            start_idx += split_size

        # --- Process in batches ---
        batch_size = args.batch_size
        target_ids_list = []
        cosine_scores_list = []
        for start in tqdm(range(0, num_combined_rows, batch_size), desc="Processing combined train+val batches"):
            end = min(start + batch_size, num_combined_rows)
            batch_embeddings_slice = combined_split_embeddings[start:end]
            batch_indices_slice = all_target_indices[start:end]
            t_ids, scores = process_batch(batch_embeddings_slice, batch_indices_slice, combined_unique_ids)
            target_ids_list.extend(t_ids)
            cosine_scores_list.extend(scores)

        # --- Build final dataset with split column ---
        final_ds = Dataset.from_dict({
            "unique_id": combined_unique_ids,
            "split": split_column,
            "target_ids": target_ids_list,
            "cosine_scores": cosine_scores_list
        })

        # --- Save combined dataset ---
        output_path = os.path.join(args.output_dir, split_name)
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving combined dataset to {output_path}")
        final_ds.save_to_disk(output_path)

        # Clean up
        del combined_split_embeddings, combined_unique_ids, combined_split_ids, split_column, all_target_indices, target_ids_list, cosine_scores_list, final_ds
        gc.collect()

    print("\nAll splits processed successfully!")
