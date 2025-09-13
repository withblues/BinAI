import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModel, AutoTokenizer
import os
import argparse
import multiprocessing as mp
from datasets import load_from_disk

class PreTrainedModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_len: int,
    ):
        self.device = device
        self.asm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.asm_tokenizer.model_max_length = max_len

        self.asm_encoder = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).to(device)

        self.asm_encoder.eval()

        self.device = device

    def forward(self, batch):
        asm_input = self.asm_tokenizer(
            batch, return_tensors="pt",
        ).to(self.device)
        asm_embeddings = self.asm_encoder(**asm_input)

        # convert to numpy for saving
        asm_embeddings_np = asm_embeddings.cpu().numpy()

        return asm_embeddings_np



def compute_embeddings(batch, model):
    combined_inputs = [
        {k: instr for k, instr in zip(keys, instrs)}
        for keys, instrs in zip(batch["keys"], batch["instructions"])
    ]

    with torch.no_grad():
        embds = model.forward(combined_inputs)

    return {"clap_embedding": embds}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_len", default=1024, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join(args.data_dir, 'assembly_x64')
    dataset = load_from_disk(dataset_path)

    # load model
    teacher_model = PreTrainedModel("hustcw/clap-asm", device, args.max_len)

    # 1. Define a dedicated directory for cache files using an absolute path.
    cache_dir = os.path.join(args.data_dir, ".cache")
    
    # 2. Create the directory yourself before calling .map(). This is the most robust way.
    os.makedirs(cache_dir, exist_ok=True)

    # 3. Create the full, unambiguous path to the cache file.
    cache_file_path = os.path.join(cache_dir, f"clap_embeddings_{args.max_len}.arrow")

    # create embeddings
    dataset = dataset.map(compute_embeddings, batched=True, batch_size=256, cache_file_name=cache_file_path, fn_kwargs={"model": teacher_model})

    dataset.save_to_disk(dataset_path + f'_{args.max_len}')
    print("precomputing done")
