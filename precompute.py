import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from models.dataset import PrecomputeDataset
from utils.data import load_data
import argparse
import pickle



class PreTrainedModel:
    def __init__(
        self,
        model_path: str,
        device: str,
    ):

        self.device = device
        self.asm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.asm_encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--indexing', action='store_true')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = args.data_dir

    # load data
    data = load_data(data_dir)

    # index data since sets are not ordered
    if args.indexing:
        data = {i: element for i, element in enumerate(data)}

        parts = data_dir.split('.')
        with open(f'{parts[0]}-indexed.{parts[1]}', 'wb') as f:
            pickle.dump(data, f)
    
    if not isinstance(data, dict):
        raise TypeError(f'Expected dict, got {type(data).__name__}. Use --indexing argument')

    # prepare dataset for CLAP
    for key in tqdm(data, desc="preparing instructions ..."):
        instructions = data[key]

        formatted_instruction = {str(i+1): inst.strip() for i, inst in enumerate(instructions)}
        data[key] = formatted_instruction
    
    # prepare dataloader to format [keys], [instructions]
    def collate_fn(batch):
        keys = [item[0] for item in batch]
        instructions = [item[1] for item in batch]
        return keys, instructions
    
    dataset = PrecomputeDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # load model
    teacher_model = PreTrainedModel('hustcw/clap-asm', device)

    output_data = {}
    with torch.no_grad():
        for data in tqdm(dataloader, desc='creating embeddings ...'):
            keys, instructions = data

            asm_input = teacher_model.asm_tokenizer(instructions, padding=True, return_tensors='pt').to(device)
            asm_embeddings = teacher_model.asm_encoder(**asm_input)

            # convert to numpy for saving
            asm_embeddings_np = asm_embeddings.cpu().numpy()

            # map key with embeddings
            for i, key in enumerate(keys):
                output_data[key.item() if isinstance(key, torch.Tensor) else key] = asm_embeddings_np[i]
    

    # save embeddings
    with open(args.output_dir, 'wb') as f:
        pickle.dump(output_data, f)

    print('precomputing done')