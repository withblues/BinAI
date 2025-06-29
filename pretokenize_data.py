import argparse
from tqdm import tqdm
import os
from src.utils.data import load_data
from src.models.tokenizer import AsmTokenizer
import torch
import numpy as np
from datasets import Dataset, Features, Value, Sequence, Array2D



def simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len):
    # does the same as BERTDataset but removes masking and stacks all combines
    # everything so that we have on instruction for each function
    student_inputs = []

    for t1_char_str, t2_char_str in data_pairs:

        # tokenizer assembly code
        t1_tokens = tokenizer.encode(t1_char_str)
        t2_tokens = tokenizer.encode(t2_char_str)

        # skip masking
        t1_padded = t1_tokens[:seq_len] + [tokenizer.vocab['[PAD]']] * (seq_len - len(t1_tokens))
        t2_padded = t2_tokens[:seq_len] + [tokenizer.vocab['[PAD]']] * (seq_len - len(t2_tokens))

        # adding CLS and SEP tokens
        bert_input = [tokenizer.vocab['[CLS]']] + t1_padded + [tokenizer.vocab['[SEP]']] + t2_padded + [tokenizer.vocab['[SEP]']]
        
        student_inputs.append(
            bert_input
        )

    return np.array(student_inputs, dtype=np.int16)
    
 
def load_assembly_data(instructions):
    pairs = []
    for i in range(0, len(instructions) - 1, 2):
        pairs.append((instructions[i].strip(), instructions[i + 1].strip()))
    return pairs


def prepare_dataset_for_bert(data, tokenizer, max_len, seq_len, total_seq_length, output_path):
    # init placeholder
    raw_dataset_features = Features({
        'function_idx': Value('int32'),
        'raw_instructions': [Value('string')] 
    })

    # helper function to load dataset to hugginface Dataset format
    def generate_raw_examples():
        sorted_indices = sorted(data.keys())
        for idx in tqdm(sorted_indices, desc='Preparing raw data for Hugging Face Dataset'):
            instructions = data[idx]
            
            yield {
                'function_idx': idx,
                'raw_instructions': instructions # List of raw assembly instruction strings
            }

    raw_hf_dataset = Dataset.from_generator(
        generate_raw_examples,
        features=raw_dataset_features,
        keep_in_memory=True
    )
    print(f"Raw dataset created with {len(raw_hf_dataset)} examples.")

    # tokenize
    def tokenize(examples):
        processed_student_inputs = []
        original_function_indices = []

        for i, instructions_list in enumerate(examples['raw_instructions']):
            function_idx = examples['function_idx'][i]
            data_pairs = load_assembly_data(instructions_list)

            data_pairs_tokenize = simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len)

            # truncate
            if len(data_pairs_tokenize) > max_len:
                final_function_pairs = data_pairs_tokenize[:max_len]
            else:
                final_function_pairs = data_pairs_tokenize
            
            processed_student_inputs.append(final_function_pairs)
            original_function_indices.append(function_idx)

        return {
            'input_ids': processed_student_inputs,
            'function_idx': original_function_indices
        }

    tokenized_features = Features({
        'function_idx': Value('int64'),
        'input_ids': Sequence(
            feature=Sequence(Value('int16'), length=total_seq_length),
            length=-1
        )
    })

    print("Tokenizing and preparing dataset with .map(). This may take a while...")
    num_processes = os.cpu_count() or 4
    
    tokenized_hf_dataset = raw_hf_dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=num_processes,
        remove_columns=['raw_instructions'],
        features=tokenized_features, 
    )

    print(f"Tokenization complete. Tokenized dataset has {len(tokenized_hf_dataset)} examples.")

    tokenized_hf_dataset.save_to_disk(output_path)


     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--max_len', default=128)
    parser.add_argument('--seq_len', default=16)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    split = args.split

    # parameters for tokenizing
    max_len = args.max_len
    seq_len = args.seq_len
    # seq_len * 2 (data pairs) + [CLS] + [SEQ] + [SEQ]
    total_seq_len = seq_len * 2 + 3

    # load input data for student model
    data = load_data(os.path.join(data_dir, f'baseline-{split}-indexed.pkl'))
    if not isinstance(data, dict):
        raise TypeError('Expected both train_data and valid_data to be dicts')
    
    # load tokenizer
    tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, f"baseline-vocab.txt"))
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # output_path
    output_path = os.path.join(output_dir, 'distil/datasets',f'{split}-tokenized')
    os.makedirs(output_path, exist_ok=True)

    prepare_dataset_for_bert(data, tokenizer, max_len, seq_len, total_seq_len, output_path)

    print(f"created dataset in {output_path}")

