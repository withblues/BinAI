import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModel, AutoTokenizer
import os
import argparse
import multiprocessing as mp
from datasets import load_from_disk


models = {
    "clap":       "hustcw/clap-asm",
    "starcoder2": "/home/wang/Data/llms/starcoder2-7b",
    "deepseek":   "/home/wang/Data/llms/deepseek-coder-7b-base-v1.5",
    "qwen":       "/home/wang/Data/llms/Qwen2.5-Coder-7B",
    "codellama":  "/home/wang/Data/llms/CodeLlama-7b-hf",
}

class PreTrainedModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_len: int,
    ):
        self.device = device
        self.asm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        if self.asm_tokenizer.pad_token is None:
            self.asm_tokenizer.pad_token = self.asm_tokenizer.eos_token
            self.asm_tokenizer.pad_token_id = self.asm_tokenizer.eos_token_id

        self.asm_tokenizer.model_max_length = max_len
        dtype = torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
        self.asm_encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, torch_dtype=dtype).to(device)

        self.asm_encoder.eval()

        self.device = device

    @torch.inference_mode()
    def forward(self, batch):
        asm_input = self.asm_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        asm_embeddings = self.asm_encoder(**asm_input)
        # get pooled results from other models
        if isinstance(asm_embeddings, torch.Tensor):
            if asm_embeddings.ndim == 2:
                pooled_asm_embeddings = asm_embeddings
            elif asm_embeddings.ndim == 3:
                mask = asm_input["attention_mask"].unsqueeze(-1)
                pooled_asm_embeddings = (asm_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unexpected tensor shape: {tuple(asm_embeddings.shape)}")
        else:
            # returns modeloutput instead of an embedding tensor
            if hasattr(asm_embeddings, "sentence_embedding") and asm_embeddings.sentence_embedding is not None:
                pooled_asm_embeddings = asm_embeddings.sentence_embedding
            elif hasattr(asm_embeddings, "pooler_output") and asm_embeddings.pooler_output is not None:
                pooled_asm_embeddings = asm_embeddings.pooler_output
            else:
                last = asm_embeddings.last_hidden_state
                mask = asm_input["attention_mask"].unsqueeze(-1)
                pooled_asm_embeddings = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # convert to numpy for saving
        return pooled_asm_embeddings.float().cpu().numpy()



def compute_embeddings(batch, model, model_name="clap"):
    if(model_name=="clap"):
        combined_inputs = [
            {k: instr for k, instr in zip(keys, instrs)}
            for keys, instrs in zip(batch["keys"], batch["instructions"])
        ]

    else:
        instrs = batch["instructions"]
        if not (isinstance(instrs[0], list) and isinstance(instrs[0][0], str)):
            raise Exception("instrs is not a batch of a list of strings!")
        else:
            combined_inputs = ['\n'.join(i) for i in instrs]
            # print(combined_inputs[0])
    embds = model.forward(combined_inputs)

    return {f"{model_name}_embedding": embds}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", default="/mnt/ambrym2/datasets/distil")
    parser.add_argument("--max_len", default=1024, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model", default="clap", type=str.lower, choices=["clap", "starcoder2", "deepseek", "qwen", "codellama"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join(args.data_dir, 'assembly_x64')
    dataset = load_from_disk(dataset_path)

    # load model
    teacher_model = PreTrainedModel(models[args.model], device, args.max_len)

    # 1. Define a dedicated directory for cache files using an absolute path.
    cache_dir = os.path.join(args.data_dir, ".cache")

    # 2. Create the directory yourself before calling .map(). This is the most robust way.
    os.makedirs(cache_dir, exist_ok=True)

    # 3. Create the full, unambiguous path to the cache file.
    cache_file_path = os.path.join(cache_dir, f"{args.model}_embeddings_{args.max_len}.arrow")

    # create embeddings
    dataset = dataset.map(compute_embeddings, batched=True, batch_size=args.batch_size, cache_file_name=cache_file_path, fn_kwargs={"model": teacher_model, "model_name": args.model})

    dataset.save_to_disk(dataset_path + f'_{args.max_len}_{args.model}')
    print("precomputing done")
