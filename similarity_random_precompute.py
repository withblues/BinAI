import argparse
from utils.data import load_data
import os
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import gc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--split', default="train")
    parser.add_argument('--data_dir', default='outputs')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4096)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_dir = args.output_dir

    np.random.seed(42)

    # load precomputed clap embeddings
    data = load_data(os.path.join(data_dir, 'clap/datasets',f'{args.split}-embeddings.pkl'))

    keys = list(data.keys())  # or your unique id field
    embeddings = np.stack([data[k] for k in keys]).astype('float32')

    function_pool = []
    total = len(embeddings)

    for start_idx in tqdm(range(0, total, args.batch_size), desc="calculating similarity ..."):
        batch_embeddings = embeddings[start_idx:start_idx + args.batch_size]
        batch_size = batch_embeddings.shape[0]

        # sample random targets
        targets = np.empty((batch_size, args.top_k), dtype=int)
        for i in range(batch_size):
            anchor_global_idx = start_idx + i

            current_anchor_targets = set()
            while len(current_anchor_targets) < args.top_k:
                rand_idx = np.random.randint(0, total)
                
                # avoid using anchor as target
                if rand_idx != anchor_global_idx: 
                    current_anchor_targets.add(rand_idx)

            targets[i] = np.array(list(current_anchor_targets)) 

        # get target embeddings
        sampled_embeddings = embeddings[targets]  

        # cosine similarity
        cosine_scores = np.einsum('bd,bkd->bk', batch_embeddings, sampled_embeddings)

        for i in range(batch_size):
            anchor_key = int(keys[start_idx + i])
            target_ids = [int(keys[idx]) for idx in targets[i]]
            scores = cosine_scores[i].tolist()

            function_pool.append({
                "anchor_id": anchor_key,
                "target_ids": target_ids,
                "cosine_scores": scores
            })


        gc.collect()

    dataset = Dataset.from_list(function_pool)

    output_path = os.path.join(output_dir, 'clap/datasets',f"{args.split}-function-pool")
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)

    print(f'created dataset at {output_path}')