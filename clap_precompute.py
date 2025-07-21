import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from src.models.dataset import PrecomputeDataset
from src.utils.data import load_data
import argparse
import pickle


class PreTrainedModel:
    def __init__(
        self,
        model_path: str,
        device: str,
    ):
        self.device = device
        self.asm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.asm_encoder = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--indexing", action="store_true")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir
    data_dir = os.path.join(data_dir, f"baseline-{args.split}.pkl")

    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, "clap/datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f"{args.split}-embeddings.pkl")

    # index data since sets are not ordered
    if args.indexing:
        data = load_data(data_dir)
        data = {i: element for i, element in enumerate(data)}

        base, ext = os.path.splitext(data_dir)
        with open(f"{base}-indexed.{ext}", "wb") as f:
            pickle.dump(data, f)

    else:
        # load data
        data = load_data(data_dir)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dict, got {type(data).__name__}. Use --indexing argument"
        )

    # prepare dataset for CLAP
    for key in tqdm(data, desc="preparing instructions ..."):
        instructions = data[key]

        formatted_instruction = {
            str(i + 1): inst.strip() for i, inst in enumerate(instructions)
        }
        data[key] = formatted_instruction

    # prepare dataloader to format [keys], [instructions]
    def collate_fn(batch):
        keys = [item[0] for item in batch]
        instructions = [item[1] for item in batch]
        return keys, instructions

    dataset = PrecomputeDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # load model
    teacher_model = PreTrainedModel("hustcw/clap-asm", device)

    output_data = {}

    # add checkpointing
    base, ext = os.path.splitext(output_dir)
    checkpoint_path = f"{base}-checkpoint.{ext}"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            output_data = pickle.load(f)

    with torch.no_grad():
        for batch_idx, data in enumerate(
            tqdm(dataloader, desc="creating embeddings ...")
        ):
            keys, instructions = data

            # skip batch if already in checkpoint
            if all(key in output_data for key in keys):
                continue

            asm_input = teacher_model.asm_tokenizer(
                instructions, padding=True, return_tensors="pt"
            ).to(device)
            asm_embeddings = teacher_model.asm_encoder(**asm_input)

            # convert to numpy for saving
            asm_embeddings_np = asm_embeddings.cpu().numpy()

            # map key with embeddings
            for i, key in enumerate(keys):
                output_data[key] = asm_embeddings_np[i]

            # checkpoint
            if (batch_idx + 1) % 1000 == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(output_data, f)
                print(f"Checkpoint saved at batch {batch_idx}")

    # save embeddings
    print(f"saving embeddings at {output_dir}")
    with open(output_dir, "wb") as f:
        pickle.dump(output_data, f)

    # remove uneccessary checkpoint
    print(f"deleting old checkpoint {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("precomputing done")
