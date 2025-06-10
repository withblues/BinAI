import argparse
from tqdm import tqdm
import os
from utils.data import load_data
from models.tokenizer import AsmTokenizer
import torch
from models.dataset import DistillDatasetTruncPadParts

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
            torch.tensor(bert_input, dtype=torch.long)
        )

    return student_inputs
    

def load_assembly_data(data):
    data_pairs = []
    for item in data:
        for i in range(0, len(item)-1, 2):
            data_pairs.append((item[i].strip(), item[i+1].strip()))

    return data_pairs

def prepare_dataset_for_bert(data, tokenizer, max_len, seq_len, total_seq_length):
    processed_data = []

    for idx, (instructions, teacher_embeddings) in tqdm(data.items(), desc='tokenize data'):
        data_pairs = load_assembly_data(instructions)

        # tokenizer each pair
        data_pairs_tokenize = simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len)

        # get length of our function
        len_function = len(data_pairs_tokenize)

        # truncate if function is longer than max_len
        final_function = []
        final_attention_mask = []
        if len_function >= max_len:
            final_function = data_pairs_tokenize[:max_len]
            final_attention_mask = [1] * max_len
        
        else: # else we pad so every function has the same length
            final_function = data_pairs_tokenize
            final_attention_mask = [1] * len_function

            # create dummy instructions
            num_padding_needed = max_len - len_function
            padded_tensor = torch.full((total_seq_length,), tokenizer.vocab['[PAD]'], dtype=torch.long)

            # add dummy instructions
            final_function.extend([padded_tensor] * num_padding_needed)
            final_attention_mask.extend([0] * num_padding_needed)

        # add to output
        processed_data.append({
            'function_id': idx,
            'student_instruction': final_function,
            'student_attention_mask': torch.tensor(final_attention_mask, dtype=torch.long),
            'teacher_embedding': torch.from_numpy(teacher_embeddings).float()
        })

    return processed_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--max_len', default=256)
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
    
    # load precomputed data from teacher model
    embedding_data = load_data(os.path.join(data_dir, f'clap-{split}-embeddings.pkl'))
    
    # concatinate dataset
    data_combined = {
        key: (data[key], embedding_data[key])
        for key in data
        if key in embedding_data
    }

    # load tokenizer
    tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, f"baseline-vocab.txt"))
    print(f"Vocab size: {len(tokenizer.vocab)}")

    tokenized_data = prepare_dataset_for_bert(data_combined, tokenizer, max_len, seq_len, total_seq_len)

    dataset = DistillDatasetTruncPadParts(tokenized_data)
    
    torch.save(tokenized_data, os.path.join(output_dir, f"distil-{split}-tokenized.pkl"))
    print(f"saved distil-{split}-tokenized.txt in {output_dir}")