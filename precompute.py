import argparse
import os

import torch
from datasets import load_from_disk
from transformers import AutoModel, AutoTokenizer


MODEL_PATHS = {
    "clap": "hustcw/clap-asm",
    "nova": "lt-asset/nova-1.3b",
}


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings while excluding padding tokens."""
    expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1)
    return summed_embeddings / token_counts


class TeacherModel:
    def __init__(self, model_path: str, device: str, max_len: int):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.model_max_length = max_len
        dtype = (
            torch.float16
            if device.startswith("cuda") and torch.cuda.is_available()
            else torch.float32
        )
        self.encoder = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        self.encoder.eval()

    @torch.inference_mode()
    def encode(self, assembly):
        model_inputs = self.tokenizer(
            assembly,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        outputs = self.encoder(**model_inputs)

        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 2:
                pooled_embeddings = outputs
            elif outputs.ndim == 3:
                pooled_embeddings = mean_pool(outputs, model_inputs["attention_mask"])
            else:
                raise ValueError(f"Unexpected model output shape: {tuple(outputs.shape)}")
        elif getattr(outputs, "sentence_embedding", None) is not None:
            pooled_embeddings = outputs.sentence_embedding
        elif getattr(outputs, "pooler_output", None) is not None:
            pooled_embeddings = outputs.pooler_output
        else:
            pooled_embeddings = mean_pool(
                outputs.last_hidden_state,
                model_inputs["attention_mask"],
            )

        return pooled_embeddings.float().cpu().numpy()


def format_assembly(batch, model_name: str):
    if model_name == "clap":
        return [
            {key: instruction for key, instruction in zip(keys, instructions)}
            for keys, instructions in zip(batch["keys"], batch["instructions"])
        ]

    instructions = batch["instructions"]
    if instructions and not (
        isinstance(instructions[0], list)
        and (not instructions[0] or isinstance(instructions[0][0], str))
    ):
        raise ValueError("Expected each function's instructions to be a list of strings")
    return ["\n".join(function_instructions) for function_instructions in instructions]


def compute_embeddings(batch, model: TeacherModel, model_name: str):
    assembly = format_assembly(batch, model_name)
    return {f"{model_name}_embedding": model.encode(assembly)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add CLAP or Nova teacher embeddings to the assembly dataset."
    )
    parser.add_argument("--data_dir", default="/mnt/ambrym2/datasets/distil")
    parser.add_argument("--max_len", default=1024, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--model",
        default="clap",
        type=str.lower,
        choices=sorted(MODEL_PATHS),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join(args.data_dir, "assembly_x64")
    dataset = load_from_disk(dataset_path)

    teacher_model = TeacherModel(MODEL_PATHS[args.model], device, args.max_len)

    cache_dir = os.path.join(args.data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(
        cache_dir,
        f"{args.model}_embeddings_{args.max_len}.arrow",
    )

    dataset = dataset.map(
        compute_embeddings,
        batched=True,
        batch_size=args.batch_size,
        cache_file_name=cache_file_path,
        fn_kwargs={"model": teacher_model, "model_name": args.model},
    )

    output_path = f"{dataset_path}_{args.max_len}_{args.model}"
    dataset.save_to_disk(output_path)
    print(f"Saved precomputed dataset to {output_path}")


if __name__ == "__main__":
    main()
