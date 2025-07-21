import argparse
from tqdm import tqdm
import os
import torch
import numpy as np
from models.tokenizer import AsmTokenizer
from utils.data import load_data


def load_assembly_data(instructions):
    pairs = []
    for i in range(0, len(instructions) - 1, 2):
        pairs.append((instructions[i].strip(), instructions[i + 1].strip()))
    return pairs


# This function remains the same as your real one, but we only need to call it once per function
def simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len):
    bert_inputs = []
    if not data_pairs:
        return torch.empty(0)  # Return empty tensor if no pairs
    for t1_char_str, t2_char_str in data_pairs:
        t1_tokens = tokenizer.encode(t1_char_str)
        t2_tokens = tokenizer.encode(t2_char_str)
        t1_padded = t1_tokens[:seq_len] + [tokenizer.vocab["[PAD]"]] * (
            seq_len - len(t1_tokens)
        )
        t2_padded = t2_tokens[:seq_len] + [tokenizer.vocab["[PAD]"]] * (
            seq_len - len(t2_tokens)
        )
        bert_input = (
            [tokenizer.vocab["[CLS]"]]
            + t1_padded
            + [tokenizer.vocab["[SEP]"]]
            + t2_padded
            + [tokenizer.vocab["[SEP]"]]
        )
        bert_inputs.append(bert_input)
    # The dtype doesn't matter for length analysis, but let's be consistent
    return torch.tensor(bert_inputs, dtype=torch.short)


def analyze_tokenized_lengths(data, tokenizer, seq_len):
    """
    CORRECTED: Analyzes the number of processed instruction tensors per function.
    """
    all_lengths = []
    data_items = list(data.items())

    for idx, instructions in tqdm(
        data_items, desc="Analyzing tokenized function lengths"
    ):
        # Step 1: Get raw instruction pairs
        data_pairs = load_assembly_data(instructions)

        # Step 2: Tokenize the pairs into BERT inputs
        # We don't need the full tensor, just its length (number of rows)
        if data_pairs:
            tokenized_tensor = simulate_BERTDataset_without_masking(
                data_pairs, tokenizer, seq_len
            )
            num_bert_inputs = tokenized_tensor.shape[0]
        else:
            num_bert_inputs = 0

        all_lengths.append(num_bert_inputs)

    return all_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tokenized function lengths")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq_len", type=int, default=16)  # Add seq_len as an argument
    args = parser.parse_args()

    # --- Load your real data and tokenizer ---
    data = load_data(os.path.join(args.data_dir, f"baseline-{args.split}-indexed.pkl"))
    tokenizer = AsmTokenizer(
        vocab_file=os.path.join(args.data_dir, f"baseline-vocab.txt")
    )

    lengths = analyze_tokenized_lengths(data, tokenizer, args.seq_len)

    # --- Print the critical statistics ---
    print("\n--- Corrected Tokenized Function Length Analysis ---")

    lengths_np = np.array(lengths)

    print(f"Total functions analyzed: {len(lengths_np)}")
    print(f"Minimum length: {np.min(lengths_np)}")
    print(f"Maximum length (The Outlier!): {np.max(lengths_np)}")
    print(f"Mean length: {np.mean(lengths_np):.2f}")
    print(f"Median length (50th percentile): {np.median(lengths_np)}")

    print("\n--- Percentiles (This is the most important part!) ---")
    print(
        f"90th percentile: {np.percentile(lengths_np, 90):.0f} (90% of functions are shorter than this)"
    )
    print(f"95th percentile: {np.percentile(lengths_np, 95):.0f}")
    print(f"98th percentile: {np.percentile(lengths_np, 98):.0f}")
    print(f"99th percentile: {np.percentile(lengths_np, 99):.0f}")
    print(f"99.5th percentile: {np.percentile(lengths_np, 99.5):.0f}")

    # --- Optional: Plotting code remains the same ---
    # ...
