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
    batch_embeddings_slice = split_embeddings[start:end]  # (batch_size, dim)
    batch_indices_slice = target_indices[start:end]       # (batch_size, top_k)
    target_embeddings_slice = split_embeddings[batch_indices_slice]  # (batch_size, top_k, dim)
    cosine_scores = np.einsum('bd,bkd->bk', batch_embeddings_slice, target_embeddings_slice)

    # map indices to unique_ids
    target_ids = np.array([[split_ids_list[idx] for idx in row] for row in batch_indices], dtype=object)
    return target_ids.tolist(), cosine_scores.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random negative pools for embeddings.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/cosine_random_pool")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    # --- Load full dataset ---
    print(f"Loading full dataset from {args.data_dir}")
    dataset = load_from_disk(args.data_dir)
    if "clap_embedding" in dataset.column_names:
        dataset = dataset.rename_column("clap_embedding", "embedding")

    # Materialize unique IDs and embeddings
    all_ids = list(dataset["unique_id"])
    dataset.set_format("numpy", columns=["embedding"])
    embeddings = dataset[:]["embedding"]
    num_rows, dim = embeddings.shape
    print(f"Embeddings shape: {embeddings.shape}")

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

        final_unique_ids = []
        final_split_column = []
        final_target_ids = []
        final_cosine_scores = []

        for data_split in ["train", "val"]:
            if data_split not in split_ids_by_group:
                continue

            split_ids = split_ids_by_group[data_split]
            split_indices = [id_to_index[uid] for uid in split_ids]
            split_embeddings = embeddings[split_indices]
            split_unique_ids = [all_ids[i] for i in split_indices]

            # --- Generate random target indices within this split only ---
            target_indices = generate_random_indices(len(split_indices), args.top_k)

            # --- Process in batches ---
            batch_size = args.batch_size
            for start in tqdm(range(0, len(split_indices), batch_size), desc=f"Processing {data_split} batches"):
                end = min(start + batch_size, len(split_indices))
                # batch_embeddings_slice = split_embeddings[start:end]
                # batch_indices_slice = target_indices[start:end]
                t_ids, scores = process_batch(split_embeddings, target_indices[start:end], split_unique_ids)
                final_target_ids.extend(t_ids)
                final_cosine_scores.extend(scores)

            final_unique_ids.extend(split_unique_ids)
            final_split_column.extend([data_split] * len(split_unique_ids))

        # --- Build final dataset ---
        final_ds = Dataset.from_dict({
            "unique_id": final_unique_ids,
            "split": final_split_column,
            "target_ids": final_target_ids,
            "cosine_scores": final_cosine_scores
        })

        # --- Save combined dataset ---
        output_path = os.path.join(args.output_dir, split_name)
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving combined dataset to {output_path}")
        final_ds.save_to_disk(output_path)

        # Clean up
        del final_ds, final_unique_ids, final_split_column, final_target_ids, final_cosine_scores
        gc.collect()

    print("\nAll splits processed successfully!")
