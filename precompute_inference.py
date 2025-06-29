import argparse
import pandas as pd
import os
from tqdm import tqdm
from src.utils.data import load_data
from datasets import Dataset
from src.utils.model import EncoderModel
import torch
import json
import time


def load_assembly_data(instructions):
    pairs = []
    for i in range(0, len(instructions) - 1, 2):
        pairs.append((instructions[i].strip(), instructions[i + 1].strip()))
    return pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--batch_size', default=64, type=int)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    model_type = args.model_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(data_dir, 'function_pools.csv'))
    unique_keys = set()

    def get_version_variants(v):
        v_str = str(v)
        if v_str.endswith('.0'):
            base = v_str[:-2]  # remove '.0'
            if base == '5':
                return [v_str, base]  # try both '5.0' and '5'
            else:
                return [base]  # e.g., '7.0' -> '7'
        else:
            return [v_str]


    for _, row in tqdm(df.iterrows(), desc='collecting unique keys ...'):
        # Get possible versions for anchor and target
        anchor_versions = get_version_variants(row["anchor_version"])
        target_versions = get_version_variants(row["target_version"])

        # Add all possible variant keys
        for av in anchor_versions:
            anchor_key = (row["anchor_function_name"], row["anchor_compiler"], av, row["anchor_opt"], row["anchor_function_bin"])
            unique_keys.add(anchor_key)
        for tv in target_versions:
            target_key = (row["target_function_name"], row["target_compiler"], tv, row["target_opt"], row["target_function_bin"])
            unique_keys.add(target_key)

    # load test data
    test_data = load_data(os.path.join(data_dir, 'baseline-test.pkl'))

    # load model
    encoder_model = EncoderModel(
        model_type=model_type,
        device=device,
        data_dir=data_dir,
        seq_length=16,
        max_len=128
    )

    embedding_dict = {}
    batch = []
    batch_keys = []
    vram_usages = []
    start_time = time.time()
    for key in tqdm(unique_keys, desc='creating batch ...'):
        try:
            asm = test_data[key]
        except KeyError:
            print(f"Warning: missing anchor function: {key}")
            continue

        # preprocess for clap or bert
        if model_type == 'clap':
            formatted_instruction = {str(i+1): inst.strip() for i, inst in enumerate(asm)}
        else:
            formatted_instruction = load_assembly_data(asm)

        batch.append(formatted_instruction)
        batch_keys.append(key)
 
        if len(batch) >= args.batch_size:
            # vram profiling
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated(device)

            embeddings = encoder_model.compute_embeddings(batch)

            mem_after = torch.cuda.memory_allocated(device)
            vram_usages.append(mem_after - mem_before)

            # save embeddings
            for k, emb in zip(batch_keys, embeddings):
                embedding_dict[k] = emb
            batch = []
            batch_keys = []

    # handle last batch
    if batch:
        # profiling
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated(device)

        embeddings = encoder_model.compute_embeddings(batch)

        mem_after = torch.cuda.memory_allocated(device)
        vram_usages.append(mem_after - mem_before)

        for k, emb in zip(batch_keys, embeddings):
            embedding_dict[k] = emb

    
    # save metadata
    data = {
        'time': time.time() - start_time,
        'avg_vram': sum(vram_usages) / len(vram_usages)
    }

    with open(os.path.join(output_dir, f'{model_type}-metadata.json'), 'w') as f:
        json.dump(data, f)

    # save data sa datasets
    records = []
    for key, emb in embedding_dict.items():
        records.append({
            "function_name": key[0],
            "compiler": key[1],
            "version": key[2],
            "opt": key[3],
            "bin": key[4],
            "embedding": emb.tolist()  # Convert numpy array to list
        })

    ds = Dataset.from_list(records)
    ds.save_to_disk(os.path.join(output_dir, f'{model_type}-test-embeddings'))


     
