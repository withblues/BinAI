import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorWithPadding,
)

TEACHER_MODEL_INFO = {
    "clap": {"path": "hustcw/clap-asm", "dim": 768},
    "nova": {
        "path": "lt-asset/nova-1.3b",
        "dim": 2048,
    },
}


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings while excluding padding tokens."""
    expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed_embeddings = (token_embeddings * expanded_mask).sum(dim=1)
    token_counts = expanded_mask.sum(dim=1).clamp(min=1)
    return summed_embeddings / token_counts


class TeacherModel:
    def __init__(self, model_path: str, device: str, max_len: int = 1024):
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


def collate_teacher_batch(batch):
    return {
        "keys": [item["keys"] for item in batch],
        "instructions": [item["instructions"] for item in batch],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate teacher or student embeddings for retrieval evaluation."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="project")
    parser.add_argument("--method", default="mse_distil")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--model",
        default="clap",
        choices=sorted(TEACHER_MODEL_INFO),
        help="Teacher model, or the teacher space used by a student projector.",
    )
    parser.add_argument("--is_teacher", action="store_true")
    parser.add_argument(
        "--checkpoint_dir",
        help="Directory containing student.pth and optionally projector.pth.",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Use a randomly initialized student instead of the MLM checkpoint.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum student length used by --filter_truncated.",
    )
    parser.add_argument(
        "--filter_truncated",
        action="store_true",
        help="Keep only functions shorter than --max_length under student tokenization.",
    )
    return parser.parse_args()


def load_test_dataset(args, cache_dir):
    dataset = load_from_disk(os.path.join(args.data_dir, "assembly_x64_1024_clap"))
    split_path = os.path.join(args.data_dir, f"cross_{args.split}_split.json")
    with open(split_path) as split_file:
        test_ids = set(json.load(split_file)["test"])

    filter_cache_dir = os.path.join(cache_dir, "dataset_filter")
    os.makedirs(filter_cache_dir, exist_ok=True)
    test_dataset = dataset.filter(
        lambda batch: [uid in test_ids for uid in batch["unique_id"]],
        batched=True,
        num_proc=16,
        cache_file_name=os.path.join(filter_cache_dir, f"{args.split}_test.arrow"),
    )

    if not args.filter_truncated:
        return test_dataset

    print("Filtering truncated examples based on student tokenization...")
    tokenizer = BertTokenizerFast.from_pretrained(
        os.path.join(args.data_dir, "tokenizer")
    )

    def get_length_flags(examples):
        texts = [
            f"{tokenizer.cls_token} "
            + f" {tokenizer.sep_token} ".join(instructions)
            + f" {tokenizer.sep_token}"
            for instructions in examples["instructions"]
        ]
        tokenized = tokenizer(texts, truncation=True, max_length=args.max_length)
        return {
            "keep": [
                len(input_ids) < args.max_length
                for input_ids in tokenized["input_ids"]
            ]
        }

    test_dataset = test_dataset.map(
        get_length_flags,
        batched=True,
        num_proc=16,
        desc="Checking lengths",
    )
    test_dataset = test_dataset.filter(
        lambda example: example["keep"],
        num_proc=16,
        desc="Filtering",
    )
    return test_dataset.remove_columns("keep")


def prepare_teacher(args, test_dataset, device):
    print(f"Loading teacher model: {args.model}")
    model = TeacherModel(TEACHER_MODEL_INFO[args.model]["path"], device)
    columns_to_keep = {"unique_id", "keys", "instructions"}
    columns_to_remove = [
        column
        for column in test_dataset.column_names
        if column not in columns_to_keep
    ]
    return model, None, test_dataset.remove_columns(columns_to_remove), collate_teacher_batch


def student_context_length(method: str) -> int:
    """Retain the original convention: use the first number in the method name."""
    match = re.search(r"\d+", method)
    return int(match.group()) if match else 128


def build_student_config(tokenizer):
    return BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=2048,
        max_position_embeddings=1024,
    )


def prepare_student(args, test_dataset, cache_dir, device):
    print(f"Loading student model for method: {args.method}")
    tokenizer = BertTokenizerFast.from_pretrained(
        os.path.join(args.data_dir, "tokenizer")
    )
    max_length = student_context_length(args.method)
    cache_folder = f"tokenization_{max_length}" if re.search(r"\d+", args.method) else "tokenization"
    tokenization_cache_dir = os.path.join(cache_dir, cache_folder)
    os.makedirs(tokenization_cache_dir, exist_ok=True)

    print(f"Using max_length={max_length} for tokenization.")
    print(f"Caching tokenized dataset in folder: {cache_folder}")

    def format_and_tokenize(examples):
        texts = [
            f"{tokenizer.cls_token} "
            + f" {tokenizer.sep_token} ".join(instructions)
            + f" {tokenizer.sep_token}"
            for instructions in examples["instructions"]
        ]
        return tokenizer(texts, truncation=True, max_length=max_length)

    columns_to_remove = [
        column for column in test_dataset.column_names if column != "unique_id"
    ]
    test_dataset = test_dataset.map(
        format_and_tokenize,
        batched=True,
        num_proc=32,
        remove_columns=columns_to_remove,
        desc="Tokenizing data for student model...",
        cache_file_name=os.path.join(
            tokenization_cache_dir,
            f"{args.split}_test.arrow",
        ),
    )

    projector = None
    if args.checkpoint_dir:
        print(f"Loading checkpoint from: {args.checkpoint_dir}")
        model = BertForMaskedLM(config=build_student_config(tokenizer))
        student_path = os.path.join(args.checkpoint_dir, "student.pth")
        if not os.path.exists(student_path):
            raise FileNotFoundError(f"student.pth not found in {args.checkpoint_dir}")
        model.load_state_dict(
            torch.load(
                student_path,
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )
        print(f"Loaded student weights from: {student_path}")

        projector_path = os.path.join(args.checkpoint_dir, "projector.pth")
        if os.path.exists(projector_path):
            teacher_dim = TEACHER_MODEL_INFO[args.model]["dim"]
            projector = nn.Linear(model.config.hidden_size, teacher_dim)
            projector.load_state_dict(
                torch.load(
                    projector_path,
                    weights_only=True,
                    map_location=torch.device("cpu"),
                )
            )
            projector.eval()
            print(f"Loaded projector weights from: {projector_path}")
        else:
            print("No projector.pth found; using the student representation directly.")
    elif args.from_scratch:
        print("Initializing the student model from scratch.")
        model = BertForMaskedLM(config=build_student_config(tokenizer))
    else:
        model_path = os.path.join(
            args.data_dir,
            f"bert_mlm_{args.split}",
            "best_model",
        )
        print(f"Loading the base MLM model from: {model_path}")
        model = BertForMaskedLM.from_pretrained(model_path)

    model = model.to(device)
    model.eval()
    if projector is not None:
        projector = projector.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return model, projector, test_dataset, collator


def teacher_inputs(batch, model_name: str):
    if model_name == "clap":
        return [
            {key: instruction for key, instruction in zip(keys, instructions)}
            for keys, instructions in zip(batch["keys"], batch["instructions"])
        ]
    return ["\n".join(instructions) for instructions in batch["instructions"]]


def generate_embeddings(
    test_dataset,
    data_collator,
    model,
    projector,
    args,
    device,
):
    if device.startswith("cuda"):
        from src.utils.gpu_stats import GPU

        gpu_monitor = GPU(interval=0.1)
    else:
        gpu_monitor = None

    unique_ids = test_dataset["unique_id"]
    inference_dataset = test_dataset.remove_columns("unique_id")
    data_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=device.startswith("cuda"),
    )

    if gpu_monitor is not None:
        gpu_monitor.start_measure()
    start_time = time.time()

    embeddings = None
    offset = 0
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Generating embeddings"):
            if args.is_teacher:
                batch_embeddings = model.encode(teacher_inputs(batch, args.model))
            else:
                model_inputs = {name: value.to(device) for name, value in batch.items()}
                outputs = model.bert(**model_inputs)
                pooled_embeddings = mean_pool(
                    outputs.last_hidden_state,
                    model_inputs["attention_mask"],
                )
                if projector is not None:
                    pooled_embeddings = projector(pooled_embeddings)
                batch_embeddings = (
                    F.normalize(pooled_embeddings, p=2, dim=-1)
                    .float()
                    .cpu()
                    .numpy()
                )

            batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
            if batch_embeddings.ndim != 2:
                raise ValueError(
                    f"Expected two-dimensional embeddings, got {batch_embeddings.shape}"
                )
            if embeddings is None:
                embeddings = np.empty(
                    (len(inference_dataset), batch_embeddings.shape[1]),
                    dtype=np.float32,
                )

            next_offset = offset + len(batch_embeddings)
            embeddings[offset:next_offset] = batch_embeddings
            offset = next_offset

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    if gpu_monitor is not None:
        gpu_monitor.stop_measure()

    if embeddings is None or offset != len(inference_dataset):
        raise RuntimeError(
            f"Generated {offset} embeddings for {len(inference_dataset)} examples"
        )

    run_data = {
        "time": elapsed_time,
        "peak_memory": (
            gpu_monitor.get_memory_usage(peak=True) if gpu_monitor is not None else 0.0
        ),
        "avg_memory": (
            gpu_monitor.get_memory_usage(average=True) if gpu_monitor is not None else 0.0
        ),
        "peak_util": (
            gpu_monitor.get_utilization(peak=True) * 100
            if gpu_monitor is not None
            else 0.0
        ),
        "avg_util": (
            gpu_monitor.get_utilization(average=True) * 100
            if gpu_monitor is not None
            else 0.0
        ),
    }
    return unique_ids, test_dataset.features["unique_id"], embeddings, run_data


def summarize_runs(runs):
    pure_runs = [run for run in runs if "mean_time" not in run]

    def mean_and_std(field):
        values = [run[field] for run in pure_runs]
        return float(np.mean(values)), float(np.std(values)) if len(values) > 1 else 0.0

    mean_time, std_time = mean_and_std("time")
    mean_avg_memory, std_avg_memory = mean_and_std("avg_memory")
    mean_peak_memory, std_peak_memory = mean_and_std("peak_memory")
    mean_avg_util, std_avg_util = mean_and_std("avg_util")
    mean_peak_util, std_peak_util = mean_and_std("peak_util")

    print("\n" + "-" * 20 + " Performance Summary " + "-" * 20)
    print(f"Runs: {len(pure_runs)}")
    print(f"Time (sec):            {mean_time:.2f} ± {std_time:.2f}")
    print(f"Avg Memory (MiB):      {mean_avg_memory:.2f} ± {std_avg_memory:.2f}")
    print(f"Peak Memory (MiB):     {mean_peak_memory:.2f} ± {std_peak_memory:.2f}")
    print(f"Avg GPU Util (%):      {mean_avg_util:.2f} ± {std_avg_util:.2f}")
    print(f"Peak GPU Util (%):     {mean_peak_util:.2f} ± {std_peak_util:.2f}")
    print("-" * 59)

    return {
        "summary_runs": len(pure_runs),
        "mean_time": mean_time,
        "std_time": std_time,
        "mean_avg_memory": mean_avg_memory,
        "std_avg_memory": std_avg_memory,
        "mean_peak_memory": mean_peak_memory,
        "std_peak_memory": std_peak_memory,
        "mean_avg_util": mean_avg_util,
        "std_avg_util": std_avg_util,
        "mean_peak_util": mean_peak_util,
        "std_peak_util": std_peak_util,
    }


def save_run_metadata(metadata_path, run_data):
    runs = []
    if os.path.exists(metadata_path):
        with open(metadata_path) as metadata_file:
            existing_data = json.load(metadata_file)
        if isinstance(existing_data, dict):
            runs = existing_data.get("runs", [])
        elif isinstance(existing_data, list):
            runs = existing_data

    runs.append(run_data)
    summary = summarize_runs(runs)
    with open(metadata_path, "w") as metadata_file:
        json.dump({"summary": summary, "runs": runs}, metadata_file, indent=4)


def save_embedding_dataset(
    output_path,
    unique_ids,
    unique_id_feature,
    embeddings: np.ndarray,
):
    if os.path.exists(output_path):
        print(f"Embedding dataset already exists; leaving it unchanged: {output_path}")
        return

    print(f"Creating embedding dataset with {len(embeddings):,} rows...")
    features = Features(
        {
            "unique_id": unique_id_feature,
            "embedding": Sequence(
                feature=Value("float32"),
                length=embeddings.shape[1],
            ),
        }
    )
    embedding_dataset = Dataset.from_dict(
        {
            "unique_id": unique_ids,
            "embedding": embeddings,
        },
        features=features,
    )
    embedding_dataset.save_to_disk(output_path)
    print(f"Saved embedding dataset to {output_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = os.path.join(args.data_dir, ".cache", args.model)
    os.makedirs(cache_dir, exist_ok=True)

    print(
        f"Inference on {args.model} using method '{args.method}' "
        f"on split '{args.split}'"
    )
    test_dataset = load_test_dataset(args, cache_dir)

    if args.is_teacher:
        model, projector, test_dataset, data_collator = prepare_teacher(
            args,
            test_dataset,
            device,
        )
    else:
        model, projector, test_dataset, data_collator = prepare_student(
            args,
            test_dataset,
            cache_dir,
            device,
        )

    unique_ids, unique_id_feature, embeddings, run_data = generate_embeddings(
        test_dataset,
        data_collator,
        model,
        projector,
        args,
        device,
    )
    print("Inference complete. Saving results and metadata...")

    init_suffix = "_scratch" if args.from_scratch and not args.is_teacher else ""
    output_dir = os.path.join(
        args.output_dir,
        "inference",
        "datasets",
        args.split,
        args.model,
    )
    os.makedirs(output_dir, exist_ok=True)

    metadata_prefix = args.model if args.is_teacher else f"{args.method}{init_suffix}"
    metadata_path = os.path.join(output_dir, f"{metadata_prefix}-metadata.json")
    save_run_metadata(metadata_path, run_data)

    embeddings_path = os.path.join(
        output_dir,
        f"{args.method}{init_suffix}-embeddings",
    )
    save_embedding_dataset(
        embeddings_path,
        unique_ids,
        unique_id_feature,
        embeddings,
    )
    print("Done.")


if __name__ == "__main__":
    main()
