import argparse
import os
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import gc
import faiss
from src.utils.data import load_data 
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hard Negative Mining with FAISS")
    parser.add_argument('--split', default="train")
    parser.add_argument('--data_dir', default='outputs')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--top_k', type=int, default=10, help="Total targets per anchor")
    parser.add_argument('--batch_size', type=int, default=2048, help="Anchors to process per batch")
    parser.add_argument('--num_candidates_faiss', type=int, default=512, help="Nearest neighbors to retrieve from FAISS")
    parser.add_argument('--num_hard_positives', type=int, default=1)
    parser.add_argument('--num_hard_negatives', type=int, default=5)
    parser.add_argument('--num_random_negatives', type=int, default=4)
    parser.add_argument('--hard_positive_threshold', type=float, default=0.7, help="Min score to be a hard positive")
    parser.add_argument('--hard_negative_max_score', type=float, default=0.7, help="Max score for a hard negative")
    parser.add_argument('--hard_negative_min_score', type=float, default=0.2, help="Min score for a hard negative")
    args = parser.parse_args()

    assert args.num_hard_positives + args.num_hard_negatives + args.num_random_negatives == args.top_k

    # load precomputed teacher embeddings
    teacher_embeddings_dict = load_data(os.path.join(args.data_dir, 'clap/datasets', f'{args.split}-embeddings.pkl'))
    keys = np.array(list(teacher_embeddings_dict.keys()))
    embeddings = np.stack(list(teacher_embeddings_dict.values())).astype('float32')
    dimension = embeddings.shape[1]
    total_embeddings = embeddings.shape[0]

    # build FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    function_pool = []
    np.random.seed(42)

    for start_idx in tqdm(range(0, total_embeddings, args.batch_size), desc="Processing anchors"):
        end_idx = min(start_idx + args.batch_size, total_embeddings)
        batch_anchor_indices = np.arange(start_idx, end_idx)
        batch_anchor_embeddings = embeddings[batch_anchor_indices]

        # query FAISS for candidates
        D, I = index.search(batch_anchor_embeddings, args.num_candidates_faiss + 1) # + 1 to include self

        for i, anchor_global_idx in enumerate(batch_anchor_indices):
            anchor_key = int(keys[anchor_global_idx])
            
            # separate candidates by score for more precise sampling
            positives_pool = []
            hard_negatives_pool = []
            
            for j in range(len(I[i])):
                candidate_idx = I[i][j]
                sim_score = D[i][j]
                if candidate_idx == anchor_global_idx: continue # Skip self

                # categorize candidates based on refined definitions
                if sim_score >= args.hard_positive_threshold:
                    positives_pool.append((candidate_idx, sim_score))
                elif args.hard_negative_min_score <= sim_score < args.hard_negative_max_score:
                    hard_negatives_pool.append((candidate_idx, sim_score))
            
            selected_targets = []
            
            ### sample from the pools
            # sample hard positives
            np.random.shuffle(positives_pool)
            num_to_add = min(len(positives_pool), args.num_hard_positives)
            selected_targets.extend(positives_pool[:num_to_add])
            
            # sample hard negatives
            np.random.shuffle(hard_negatives_pool)
            num_to_add = min(len(hard_negatives_pool), args.num_hard_negatives)
            selected_targets.extend(hard_negatives_pool[:num_to_add])
            
            # fill remaining with random negatives
            num_random_to_add = args.top_k - len(selected_targets)
            if num_random_to_add > 0:
                current_target_keys = {int(keys[idx]) for idx, _ in selected_targets}
                random_indices = np.random.choice(total_embeddings, size=num_random_to_add * 2, replace=False)
                
                added_count = 0
                for rand_idx in random_indices:
                    if added_count >= num_random_to_add: break
                    if rand_idx != anchor_global_idx and int(keys[rand_idx]) not in current_target_keys:
                        score = np.dot(batch_anchor_embeddings[i], embeddings[rand_idx])
                        selected_targets.append((rand_idx, score))
                        added_count += 1

            # fallback if we still don't have enough (unlikely with this method but safe)
            while len(selected_targets) < args.top_k:
                rand_idx = np.random.randint(0, total_embeddings)
                if rand_idx != anchor_global_idx:
                    score = np.dot(batch_anchor_embeddings[i], embeddings[rand_idx])
                    selected_targets.append((rand_idx, score))

            # finalize the list for this anchor
            selected_targets = selected_targets[:args.top_k] # Ensure exactly top_k
            
            # shuffle data
            final_target_ids = [int(keys[idx]) for idx, _ in selected_targets]
            final_cosine_scores = [float(score) for _, score in selected_targets]
            combined_list = list(zip(final_target_ids, final_cosine_scores))

            random.shuffle(combined_list)
            shuffled_ids, shuffled_scores = zip(*combined_list)

            function_pool.append({
                "anchor_id": anchor_key,
                "target_ids": list(shuffled_ids),      # Use the shuffled lists
                "cosine_scores": list(shuffled_scores)
            })

        gc.collect()

    dataset = Dataset.from_list(function_pool)
    output_path = os.path.join(args.output_dir, 'clap/datasets', f"{args.split}-function-pool-hard_mined")
    dataset.save_to_disk(output_path)
    print(f'Created hard-mined dataset at {output_path}')