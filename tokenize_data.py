import argparse
from tqdm import tqdm
import os
from utils.data import load_data
from models.tokenizer import AsmTokenizer
import torch
import webdataset as wds
import json

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

    return torch.tensor(student_inputs, dtype=torch.short)
    
 
def load_assembly_data(instructions):
    pairs = []
    for i in range(0, len(instructions) - 1, 2):
        pairs.append((instructions[i].strip(), instructions[i + 1].strip()))
    return pairs

def prepare_dataset_for_bert(data, tokenizer, max_len, seq_len, output_pattern):
    with wds.ShardWriter(output_pattern, maxsize=500 << 20) as sink:
        for idx, instructions in tqdm(data.items(), desc='Creating webdataset shards'):
            data_pairs = load_assembly_data(instructions)
            
            # tokenizer each pair
            data_pairs_tokenize = simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len)

            # truncate
            final_function_tensor = data_pairs_tokenize[:max_len, :]

            # create sample for webdataset
            sample = {
                "__key__": str(idx),
                "student.pth": final_function_tensor,
            }

            sink.write(sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--checkpoint_every', type=int, default=500000)
    parser.add_argument('--split', default='train')
    parser.add_argument('--max_len', default=128)
    parser.add_argument('--seq_len', default=16)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    split = args.split
    checkpoint_dir = args.checkpoint_dir
    checkpoint_every = args.checkpoint_every

    # parameters for tokenizing
    max_len = args.max_len
    seq_len = args.seq_len

    # load input data for student model
    data = load_data(os.path.join(data_dir, f'baseline-{split}-indexed.pkl'))
    if not isinstance(data, dict):
        raise TypeError('Expected both train_data and valid_data to be dicts')
    
    # load tokenizer
    tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, f"baseline-vocab.txt"))
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # create folder for shards and saving pattern
    shard_dir = os.path.join(output_dir, f'distil-{split}-tokenized-shards')
    os.makedirs(shard_dir, exist_ok=True)
    output_pattern = os.path.join(shard_dir, f'shards--%06d.tar')

    prepare_dataset_for_bert(data, tokenizer, max_len, seq_len, output_pattern)

    # save metadata to get length of dataloader
    total_samples = len(data)
    metadata_path = os.path.join(shard_dir, 'metadata.json')

    with open(metadata_path, 'w') as f:
        metadata = {'total_samples': total_samples}
        json.dump(metadata, f)

    print(f"created webdataset in {output_dir}")
    print(f' saved metadata file with {total_samples} samples to {metadata_path}')
