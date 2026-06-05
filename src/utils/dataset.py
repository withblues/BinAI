import torch
from torch.utils.data import Dataset
    
class CosineDataset(Dataset):
    def __init__(self, dataset, lookup, id2idx, top_k=None):
        self.dataset = dataset
        self.lookup = lookup
        self.id2idx = id2idx
        self.top_k = top_k

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor = self.dataset[idx]
        target_ids, _ = self.lookup[anchor["unique_id"]]
        
        # Only slice if top_k is explicitly provided
        if self.top_k is not None:
            target_ids = target_ids[:self.top_k]

        input_ids = [anchor["input_ids"]]
        attention_masks = [anchor["attention_mask"]]
        teacher_embeddings = [anchor["labels"]] # The vector from the base dataset

        for tid in target_ids:
            t_idx = self.id2idx[tid]           
            target_example = self.dataset[t_idx]
            input_ids.append(target_example["input_ids"])
            attention_masks.append(target_example["attention_mask"])
            teacher_embeddings.append(target_example["labels"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "teacher_embeddings": teacher_embeddings,
        }
    
class InfoNCEDatasetWithLookup(Dataset):
    def __init__(self, base_dataset, lookup, id2idx, top_k):
        self.base_dataset = base_dataset
        self.lookup = lookup
        self.id2idx = id2idx
        self.top_k = top_k
        self.num_negatives = self.top_k - 1
        
        # We still flatten the lookup for easy indexing
        self.flat_examples = []
        for anchor_id, examples in lookup.items():
            for ex in examples:
                self.flat_examples.append({
                    'anchor_id': anchor_id,
                    'positive_id': ex['positive_id'],
                    'negative_ids': ex['negative_ids']
                })
        
        # Get a list of all valid IDs from the id2idx map for random sampling fallback
        self.valid_ids_pool = list(self.id2idx.keys())


    def __len__(self):
        return len(self.flat_examples)

    def __getitem__(self, idx):
        example = self.flat_examples[idx]
        anchor_id = example['anchor_id']
        positive_id = example['positive_id']
        negative_ids = example['negative_ids']

        anchor_tokens = self.base_dataset[self.id2idx[anchor_id]]
        positive_tokens = self.base_dataset[self.id2idx[positive_id]]
        
        all_input_ids = [anchor_tokens["input_ids"]]
        all_attention_masks = [anchor_tokens["attention_mask"]]
        
        all_input_ids.append(positive_tokens["input_ids"])
        all_attention_masks.append(positive_tokens["attention_mask"])

        # --- THIS IS THE CORRECTED LOGIC ---
        found_negatives_tokens = []
        for neg_id in negative_ids:
            target_idx = self.id2idx.get(neg_id)
            if target_idx is not None:
                target_tokens = self.base_dataset[target_idx]
                found_negatives_tokens.append(target_tokens)
        
        # Fallback: If we found fewer negatives than required, fill with randoms
        # This guarantees the output size is always correct.
        forbidden_ids = {anchor_id, positive_id}
        while len(found_negatives_tokens) < self.num_negatives:
            # Sample a random ID from the pool of *valid* IDs for this split
            rand_id = random.choice(self.valid_ids_pool)
            if rand_id not in forbidden_ids:
                rand_idx = self.id2idx[rand_id]
                found_negatives_tokens.append(self.base_dataset[rand_idx])
                forbidden_ids.add(rand_id) # Avoid picking the same fallback twice
        
        # Add the final list of negatives
        for tokens in found_negatives_tokens[:self.num_negatives]: # Truncate just in case
            all_input_ids.append(tokens["input_ids"])
            all_attention_masks.append(tokens["attention_mask"])

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": 0,
        }