import os
import json
import numpy as np
import argparse
from datasets import load_from_disk, Dataset, DatasetDict
import gc
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process and clean a mined dataset for InfoNCE (Robust Version).")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load lookups (same as before) ---
    print("Loading original metadata and building GT lookup...")
    original_dataset = load_from_disk(args.data_dir)
    original_dataset.set_format("numpy", columns=['unique_id', 'binary_name', 'function_name'])
    all_data_np = original_dataset[:]
    metadata_lookup = {int(uid): {'binary_name': str(b_name), 'function_name': str(f_name)} for uid, b_name, f_name in zip(all_data_np['unique_id'], all_data_np['binary_name'], all_data_np['function_name'])}
    del all_data_np; gc.collect()

    # --- 2. Load the source hybrid dataset ---
    print(f"Loading hybrid dataset from: {args.input_dir}")
    hybrid_dataset = load_from_disk(args.input_dir)
    
    # --- MODIFICATION: Initialize global lists to hold all data ---
    # Instead of a dictionary of datasets, we'll build single lists.
    final_anchor_ids = []
    final_positive_ids = []
    final_negative_ids = []
    final_splits = [] # This new list will store 'train' or 'val' for each row

    for split in ['train', 'val']:
        print(f"\nProcessing '{split}' split...")
        # Note: hybrid_dataset is already a DatasetDict, so we can access splits directly
        # Using .filter is fine too, but this is slightly cleaner if splits exist.
        try:
            split_data = hybrid_dataset.filter(lambda x: x['split'] == split, num_proc=16, keep_in_memory=True)
        except KeyError:
            print(f"Split '{split}' not found in the input dataset. Skipping.")
            continue
            
        n_rows = len(split_data)
        n_unique = len(set(split_data["unique_id"]))
        print(f"{split}: rows={n_rows}, unique unique_id={n_unique}, duplicates={n_rows - n_unique}")

        # The inner loop logic remains the same
        for row in tqdm(split_data, desc=f"Cleaning and processing {split} data"):
            anchor_id = row['unique_id']
            target_ids = row['target_ids']
            anchor_meta = metadata_lookup.get(anchor_id)
            if not anchor_meta: continue
            
            positives_in_row, negatives_in_row = [], []
            for target_id in target_ids:
                target_meta = metadata_lookup.get(target_id)
                if not target_meta: continue
                if (anchor_meta['function_name'] == target_meta['function_name'] and
                    anchor_meta['binary_name'] == target_meta['binary_name']):
                    positives_in_row.append(target_id)
                else:
                    negatives_in_row.append(target_id)
            
            # --- MODIFICATION: Append to global lists ---
            if positives_in_row:
                final_anchor_ids.append(anchor_id)
                final_positive_ids.append(positives_in_row)
                final_negative_ids.append(negatives_in_row)
                final_splits.append(split) # Add the split name for this row

        print(f"Finished processing '{split}'. Added {len(final_anchor_ids) - len(final_splits) + len(split_data)} clean examples.")


    # --- 4. Combine and save the final Dataset ---
    if not final_anchor_ids:
        print("No valid data found. Exiting.")
    else:
        # --- MODIFICATION: Create a single Dataset from the combined lists ---
        # Note: Your example used "target_ids" and "cosine_scores". Your script
        # generates "positive_ids" and "negative_ids", so I'm using those.
        # If you have cosine scores in your input, you could add them here too.
        final_dataset = Dataset.from_dict({
            "unique_id": final_anchor_ids,
            "positive_ids": final_positive_ids,
            "negative_ids": final_negative_ids,
            "split": final_splits
        })

        print(f"\nFinal combined dataset stats:\n{final_dataset}")
        print(f"Saving single clean Dataset to {args.output_dir}")
        final_dataset.save_to_disk(args.output_dir)

    print("\nPost-processing complete!")