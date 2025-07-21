import pandas as pd
import os
import argparse
from tqdm import tqdm
from datasets import load_from_disk
import numpy as np


def find_embedding(embedding_dict, key):
    embedding = embedding_dict.get(key)
    if embedding is not None:
        return embedding

    name, compiler, version, opt, bin_name = key

    alternate_versions = []
    if "." in version:
        # input is '5.0', try '5'
        base_version = version.split(".")[0]
        alternate_versions.append(base_version)
    else:
        # input is '5', try '5.0'
        alternate_versions.append(f"{version}.0")

    for alt_ver in alternate_versions:
        variant_key = (name, compiler, alt_ver, opt, bin_name)
        embedding = embedding_dict.get(variant_key)
        if embedding is not None:
            return embedding

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--chunksize", default=10000, type=int, dest="chunksize")

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    model_type = args.model_type

    # load precomputed embeddings and create lookup table
    dataset = load_from_disk(
        os.path.join(data_dir, "inference/datasets", f"{model_type}-test-embeddings")
    )
    embedding_dict = {
        (
            row["function_name"],
            row["compiler"],
            row["version"],
            row["opt"],
            row["bin"],
        ): np.array(row["embedding"], dtype=np.float32)
        for row in dataset
    }

    # load csv
    df = pd.read_csv(
        os.path.join(data_dir, "function_pools.csv"),
        dtype={"anchor_version": str, "target_version": str},
    )
    df_grouped = df.groupby(
        [
            "anchor_function_name",
            "anchor_compiler",
            "anchor_version",
            "anchor_opt",
            "anchor_function_bin",
        ]
    )

    results = []
    similarity_col_name = f"sim_{model_type}"

    for anchor_key, group_df in tqdm(df_grouped, desc="processing anchors..."):
        anchor_embedding = find_embedding(embedding_dict, anchor_key)

        if anchor_embedding is None:
            print(f"key {anchor_key} not found")
            continue

        # collect taret embeddings
        target_keys = []
        target_embeddings_list = []

        for _, row in group_df.iterrows():
            target_key = (
                row["target_function_name"],
                row["target_compiler"],
                row["target_version"],
                row["target_opt"],
                row["target_function_bin"],
            )
            target_embedding = find_embedding(embedding_dict, target_key)

            if target_embedding is not None:
                target_keys.append(target_key)
                target_embeddings_list.append(target_embedding)

        if not target_embeddings_list:
            print(f"no target keys for anchor {anchor_key} found")
            continue

        # calculate cosine similarity given embeddings are normalized
        target_matrix = np.vstack(target_embeddings_list)
        similarities = np.dot(target_matrix, anchor_embedding)

        # save data to results
        (a_name, a_compiler, a_ver, a_opt, a_bin) = anchor_key
        for target_info, sim_score in zip(target_keys, similarities):
            (t_name, t_compiler, t_ver, t_opt, t_bin) = target_info
            results.append(
                {
                    "anchor_function_bin": a_bin,
                    "anchor_function_name": a_name,
                    "anchor_compiler": a_compiler,
                    "anchor_version": a_ver,
                    "anchor_opt": a_opt,
                    "target_function_bin": t_bin,
                    "target_function_name": t_name,
                    "target_compiler": t_compiler,
                    "target_version": t_ver,
                    "target_opt": t_opt,
                    similarity_col_name: sim_score,
                }
            )

    print(f"processing complete. saving {len(results)} pairs to CSV...")
    output_dir = os.path.join(output_dir, "inference/cosine_similarity")
    os.makedirs(output_dir, exist_ok=True)

    output_df = pd.DataFrame(results)
    output_df.to_csv(
        os.path.join(output_dir, f"{model_type}-results-cosine.csv"), index=False
    )
    print("Done.")
