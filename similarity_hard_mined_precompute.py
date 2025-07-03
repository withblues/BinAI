# import argparse
# import os
# import numpy as np
# from tqdm import tqdm
# from datasets import Dataset
# import gc
# import faiss # Import FAISS
# from src.utils.data import load_data # Assuming this loads your embeddings dict

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Command line parameters")
#     parser.add_argument('--split', default="train")
#     parser.add_argument('--data_dir', default='outputs')
#     parser.add_argument('--output_dir', default='outputs')
#     parser.add_argument('--top_k', type=int, default=10) # Total targets per anchor
#     #parser.add_argument('--batch_size', type=int, default=4096) # For processing anchors
#     parser.add_argument('--batch_size', type=int, default=128) # For processing anchors
#     parser.add_argument('--num_candidates_faiss', type=int, default=512) # How many nearest neighbors to retrieve from FAISS
#     parser.add_argument('--num_hard_positives', type=int, default=1) # Number of actual "hard positives" (highest sim)
#     parser.add_argument('--num_hard_negatives', type=int, default=5) # Number of "hard negatives" (moderate sim)
#     parser.add_argument('--num_random_negatives', type=int, default=4) # Number of purely random negatives
#     args = parser.parse_args()

#     # Ensure the counts add up to top_k
#     assert args.num_hard_positives + args.num_hard_negatives + args.num_random_negatives == args.top_k, \
#         f"Sum of positive, hard negative, and random negative counts must equal top_k ({args.top_k})"

    
#     data_dir = args.data_dir
#     output_dir = args.output_dir

#     np.random.seed(42)

#     # Load precomputed clap embeddings
#     # data format: {function_id: embedding_vector}
#     teacher_embeddings_dict = load_data(os.path.join(data_dir, 'clap/datasets', f'{args.split}-embeddings.pkl'))

#     # Prepare data for FAISS
#     keys = np.array(list(teacher_embeddings_dict.keys()))
#     embeddings = np.stack(list(teacher_embeddings_dict.values())).astype('float32')

#     dimension = embeddings.shape[1] # Dimension of your embeddings (e.g., 128)

#     # 1. Build FAISS Index
#     print("Building FAISS index...")
#     # A simple index type: Flat index (brute-force search, but accurate)
#     # For very large datasets, consider IVFFlat or HNSW for speed
#     index = faiss.IndexFlatIP(dimension) # IP for Inner Product, suitable for cosine similarity

#     faiss.normalize_L2(embeddings) # Normalize embeddings for cosine similarity with IP index
#     index.add(embeddings)
#     print(f"FAISS index built with {index.ntotal} vectors.")

#     function_pool = []
#     total_embeddings = embeddings.shape[0]

#     print("Calculating similarity and selecting targets...")
#     for start_idx in tqdm(range(0, args.batch_size, args.batch_size), desc="Processing anchors"):
#         batch_anchor_indices = np.arange(start_idx, min(start_idx + args.batch_size, total_embeddings))
#         batch_anchor_embeddings = embeddings[batch_anchor_indices] # Already L2-normalized

#         # 2. Query FAISS Index for candidates
#         # D: distances (inner products, which are cosine sim here), I: indices
#         # We query for `num_candidates_faiss` neighbors for each anchor in the batch
#         D, I = index.search(batch_anchor_embeddings, args.num_candidates_faiss + 1) # +1 to include self (anchor)

#         for i, anchor_global_idx in enumerate(batch_anchor_indices):
#             anchor_key = int(keys[anchor_global_idx]) # Original function ID

#             # Get candidates for this specific anchor
#             candidate_indices = I[i]
#             candidate_similarities = D[i]

#             # Filter out the anchor itself from candidates
#             filtered_candidates = []
#             for j in range(len(candidate_indices)):
#                 if candidate_indices[j] != anchor_global_idx: # Ensure it's not the anchor
#                     filtered_candidates.append((candidate_indices[j], candidate_similarities[j]))

#             # Sort by similarity (descending)
#             filtered_candidates.sort(key=lambda x: x[1], reverse=True)

#             selected_target_ids = []
#             selected_cosine_scores = []
            
#             # 3. Select Hard Positives
#             # These are the *most similar* functions according to the teacher, excluding the anchor itself.
#             # You might need to adjust the similarity threshold if your "positives" are very few or too many
#             # A "hard positive" is often another example of the *same* class or a very semantically close one.
#             # In your case, it's just the highest similarity functions.
            
#             # Collect potential positives/hard negatives first
#             potential_hard_targets = []
#             for candidate_idx, sim_score in filtered_candidates:
#                 # Add to potential hard targets up to a certain point (e.g., top N beyond hard positives)
#                 # Or based on a similarity threshold (e.g., sim_score > 0.1)
#                 potential_hard_targets.append((candidate_idx, sim_score))

#             # Ensure we have enough candidates
#             if len(potential_hard_targets) < args.num_hard_positives + args.num_hard_negatives:
#                 # Fallback: if not enough hard candidates, fill with randoms later
#                 pass # Will be handled by random negatives later if slots are left

#             # Select Hard Positives
#             num_hp_added = 0
#             for k_idx, (target_faiss_idx, score) in enumerate(potential_hard_targets):
#                 if num_hp_added >= args.num_hard_positives:
#                     break
                
#                 # You might add a minimum similarity threshold for "hard positives" if needed
#                 # e.g., if score < 0.7, don't consider it a hard positive
                
#                 selected_target_ids.append(int(keys[target_faiss_idx]))
#                 selected_cosine_scores.append(float(score))
#                 num_hp_added += 1

#             # Select Hard Negatives
#             # These are usually from the moderately similar candidates.
#             # You want those that are "close" in embedding space but should be distinct.
#             num_hn_added = 0
#             for k_idx in range(num_hp_added, len(potential_hard_targets)): # Start after hard positives
#                 if num_hn_added >= args.num_hard_negatives:
#                     break
#                 target_faiss_idx, score = potential_hard_targets[k_idx]
                
#                 # Crucial: Define what a "hard negative" means by its score range.
#                 # Example: If your "positives" are > 0.7, hard negatives might be 0.1 to 0.7
#                 # Adjust these thresholds based on your data's similarity distribution.
#                 # If your teacher is perfect, these are true "negatives" (low sim) but were retrieved as neighbors.
#                 # You might want to sample from a range like 0.1 to 0.5.
                
#                 selected_target_ids.append(int(keys[target_faiss_idx]))
#                 selected_cosine_scores.append(float(score))
#                 num_hn_added += 1

#             # Fill remaining slots with purely Random Negatives
#             num_random_to_add = args.top_k - len(selected_target_ids)
#             if num_random_to_add > 0:
#                 random_candidates_indices = np.random.choice(total_embeddings, size=num_random_to_add * 2, replace=False) # Get more than needed
#                 added_count = 0
#                 for rand_idx in random_candidates_indices:
#                     if added_count >= num_random_to_add:
#                         break
#                     # Ensure not anchor and not already selected
#                     if rand_idx != anchor_global_idx and int(keys[rand_idx]) not in selected_target_ids:
#                         target_key = int(keys[rand_idx])
                        
#                         # Calculate cosine score for this truly random one
#                         target_embedding = embeddings[rand_idx] # Already L2-normalized
#                         score = np.dot(batch_anchor_embeddings[i], target_embedding)
                        
#                         selected_target_ids.append(target_key)
#                         selected_cosine_scores.append(float(score))
#                         added_count += 1
            
#             # Ensure top_k is met (important if not enough candidates or randoms were found)
#             if len(selected_target_ids) < args.top_k:
#                 print(f"Warning: Could not find {args.top_k} targets for anchor {anchor_key}. Found {len(selected_target_ids)}. Filling with duplicates/more random.")
#                 # This could happen if your `num_candidates_faiss` is too low, or dataset too small.
#                 # For simplicity, we can duplicate existing or add more randoms.
#                 while len(selected_target_ids) < args.top_k:
#                     # Just add a random one again if we're short
#                     rand_idx = np.random.randint(0, total_embeddings)
#                     if rand_idx != anchor_global_idx:
#                         target_key = int(keys[rand_idx])
#                         target_embedding = embeddings[rand_idx]
#                         score = np.dot(batch_anchor_embeddings[i], target_embedding)
#                         selected_target_ids.append(target_key)
#                         selected_cosine_scores.append(float(score))


#             function_pool.append({
#                 "anchor_id": anchor_key,
#                 "target_ids": selected_target_ids,
#                 "cosine_scores": selected_cosine_scores
#             })

#         gc.collect()

#     dataset = Dataset.from_list(function_pool)

#     output_path = os.path.join(output_dir, 'clap/datasets',f"{args.split}-function-pool-hard_mined") 
#     os.makedirs(output_path, exist_ok=True)
#     dataset.save_to_disk(output_path)

#     print(f'Created hard-mined dataset at {output_path}')

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
    # Optional: Add thresholds to better define positive/negative
    parser.add_argument('--hard_positive_threshold', type=float, default=0.7, help="Min score to be a hard positive")
    parser.add_argument('--hard_negative_max_score', type=float, default=0.7, help="Max score for a hard negative")
    parser.add_argument('--hard_negative_min_score', type=float, default=0.2, help="Min score for a hard negative")
    args = parser.parse_args()

    assert args.num_hard_positives + args.num_hard_negatives + args.num_random_negatives == args.top_k

    # Load precomputed teacher embeddings
    teacher_embeddings_dict = load_data(os.path.join(args.data_dir, 'clap/datasets', f'{args.split}-embeddings.pkl'))
    keys = np.array(list(teacher_embeddings_dict.keys()))
    embeddings = np.stack(list(teacher_embeddings_dict.values())).astype('float32')
    dimension = embeddings.shape[1]
    total_embeddings = embeddings.shape[0]

    # 1. Build FAISS Index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    #faiss.normalize_L2(embeddings) # Must normalize for IP to be cosine similarity
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    function_pool = []
    np.random.seed(42)

    # *** CRITICAL FIX: Correct the main loop range ***
    #for start_idx in tqdm(range(0, args.batch_size, args.batch_size), desc="Processing anchors"):
    for start_idx in tqdm(range(0, total_embeddings, args.batch_size), desc="Processing anchors"):
        end_idx = min(start_idx + args.batch_size, total_embeddings)
        batch_anchor_indices = np.arange(start_idx, end_idx)
        batch_anchor_embeddings = embeddings[batch_anchor_indices]

        # 2. Query FAISS for candidates
        D, I = index.search(batch_anchor_embeddings, args.num_candidates_faiss + 1) # +1 to include self

        for i, anchor_global_idx in enumerate(batch_anchor_indices):
            anchor_key = int(keys[anchor_global_idx])
            
            # Separate candidates by score for more precise sampling
            positives_pool = []
            hard_negatives_pool = []
            
            for j in range(len(I[i])):
                candidate_idx = I[i][j]
                sim_score = D[i][j]
                if candidate_idx == anchor_global_idx: continue # Skip self

                # Categorize candidates based on refined definitions
                if sim_score >= args.hard_positive_threshold:
                    positives_pool.append((candidate_idx, sim_score))
                elif args.hard_negative_min_score <= sim_score < args.hard_negative_max_score:
                    hard_negatives_pool.append((candidate_idx, sim_score))
            
            selected_targets = []
            
            # 3. Sample from the pools
            # Sample Hard Positives
            np.random.shuffle(positives_pool)
            num_to_add = min(len(positives_pool), args.num_hard_positives)
            selected_targets.extend(positives_pool[:num_to_add])
            
            # Sample Hard Negatives
            np.random.shuffle(hard_negatives_pool)
            num_to_add = min(len(hard_negatives_pool), args.num_hard_negatives)
            selected_targets.extend(hard_negatives_pool[:num_to_add])
            
            # 4. Fill remaining with Random Negatives
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

            # Fallback if we still don't have enough (unlikely with this method but safe)
            while len(selected_targets) < args.top_k:
                rand_idx = np.random.randint(0, total_embeddings)
                if rand_idx != anchor_global_idx:
                    score = np.dot(batch_anchor_embeddings[i], embeddings[rand_idx])
                    selected_targets.append((rand_idx, score))

            # Finalize the list for this anchor
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