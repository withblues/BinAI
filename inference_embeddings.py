import argparse
import os
import torch
import json
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk, Dataset, Features, Value, Sequence
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast, BertForMaskedLM, DataCollatorWithPadding, BertConfig
from src.utils.gpu_stats import GPU
import re

# --- Merged Model Information from both scripts ---
teacher_model_info = {
    "clap":       {"path": "hustcw/clap-asm", "dim": 768},
    "starcoder2": {"path": "/mnt/ambrym2/datasets/distil/llms/starcoder2-7b", "dim": 4608},
    "deepseek":   {"path": "/mnt/ambrym2/datasets/distil/llms/deepseek-coder-7b-base-v1.5", "dim": 4096},
    "qwen":       {"path": "/mnt/ambrym2/datasets/distil/llms/Qwen2.5-Coder-7B", "dim": 3584},
    "codellama":  {"path": "/mnt/ambrym2/datasets/distil/llms/CodeLlama-7b-hf", "dim": 4096},
    "nova":      {"path": "/mnt/ambrym2/datasets/distil/llms/Novacode-7B", "dim": 2048},    
}
teacher_model_names = list(teacher_model_info.keys())


# --- Class from the second script to handle teacher LLMs ---
class PreTrainedModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_len: int = 1024,
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
        self.asm_encoder = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True, torch_dtype=dtype
        ).to(device)
        self.asm_encoder.eval()

    @torch.inference_mode()
    def forward(self, batch):
        asm_input = self.asm_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        asm_embeddings = self.asm_encoder(**asm_input)
        
        # Comprehensive pooling logic from your second script
        if isinstance(asm_embeddings, torch.Tensor):
            if asm_embeddings.ndim == 2:
                pooled_asm_embeddings = asm_embeddings
            elif asm_embeddings.ndim == 3:
                mask = asm_input["attention_mask"].unsqueeze(-1)
                pooled_asm_embeddings = (asm_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unexpected tensor shape: {tuple(asm_embeddings.shape)}")
        else:
            if hasattr(asm_embeddings, "sentence_embedding") and asm_embeddings.sentence_embedding is not None:
                pooled_asm_embeddings = asm_embeddings.sentence_embedding
            elif hasattr(asm_embeddings, "pooler_output") and asm_embeddings.pooler_output is not None:
                pooled_asm_embeddings = asm_embeddings.pooler_output
            else:
                last = asm_embeddings.last_hidden_state
                mask = asm_input["attention_mask"].unsqueeze(-1)
                pooled_asm_embeddings = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled_asm_embeddings.float().cpu().numpy()

def custom_collate_for_text(batch):
    """
    Collates a list of samples into a single batch dictionary.
    Instead of stacking, it creates lists for each key.
    """
    keys = [item['keys'] for item in batch]
    instructions = [item['instructions'] for item in batch]
    return {'keys': keys, 'instructions': instructions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--method", default='mse_distil')
    parser.add_argument("--batch_size", default=64, type=int)
    # --- Updated model choices ---
    parser.add_argument("--model", default='clap', type=str, 
                        help="Can be a teacher model or a student model config.")
    parser.add_argument("--is_teacher", action='store_true',)
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to checkpoint directory containing student.pth and optionally projector.pth.")
    parser.add_argument("--from_scratch", action='store_true', help="Initialize the student model with random weights using the default architecture instead of loading pretrained weights.")
    parser.add_argument("--max_length", type=int, default=128, help="Max context length for the model")
    parser.add_argument("--filter_truncated", action='store_true', help="Filter out any data that is equal to or exceeds student max_len tokens.")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    # Cache directory now depends on the specific model being run
    cache_dir = os.path.join(args.data_dir, ".cache", args.model)
    os.makedirs(cache_dir, exist_ok=True)
    method = args.method
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Inference on {args.model} model using method '{method}' on split '{args.split}'")

    dataset = load_from_disk(os.path.join(data_dir, 'assembly_x64_1024_clap'))
    with open(os.path.join(data_dir, f"cross_{args.split}_split.json")) as f:
        indices = json.load(f)

    test_ids = set(indices['test'])
    test_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{args.split}_test.arrow")
    test_dataset = dataset.filter(lambda batch: [uid in test_ids for uid in batch["unique_id"]], batched=True, num_proc=16, cache_file_name=test_cache_filter_path)

    if args.filter_truncated:
        print("Filtering truncated examples based on student tokenization...")
        student_tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))
        
        def get_len_flag(examples):
            texts = [
                f"{student_tokenizer.cls_token} " + f" {student_tokenizer.sep_token} ".join(instr_list) + f" {student_tokenizer.sep_token}"
                for instr_list in examples["instructions"]
            ]
            tokenized = student_tokenizer(texts, truncation=True, max_length=args.max_length)
            return {"keep": [len(ids) < args.max_length for ids in tokenized["input_ids"]]}
        
        test_dataset = test_dataset.map(get_len_flag, batched=True, num_proc=16, desc="Checking lengths")
        test_dataset = test_dataset.filter(lambda x: x["keep"], num_proc=16, desc="Filtering")
        test_dataset = test_dataset.remove_columns(["keep"])

    model = None
    tokenizer = None
    data_collator = None
    init_suffix = ""

    # --- REFACTORED LOGIC: Handle TEACHER models first ---
    if args.is_teacher:
        print(f"Loading TEACHER model: {args.model}")
        model_path = teacher_model_info[args.model]["path"]
        model = PreTrainedModel(model_path, device)
        tokenizer = model.asm_tokenizer # For reference, though not used for pre-tokenization
        data_collator = custom_collate_for_text

        columns_to_keep = ['unique_id', 'keys', 'instructions']
        columns_to_remove = [c for c in test_dataset.column_names if c not in columns_to_keep]
        test_dataset = test_dataset.remove_columns(columns_to_remove)

    # --- ORIGINAL LOGIC: Handle STUDENT models ---
    else:
        print(f"Loading STUDENT model for method: {method}")
        # Tokenizing for student BERT model
        tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))
        

        # Extract first number from the method
        match = re.search(r"\d+", method)
        max_length = int(match.group()) if match else 128

        # test_cache_folder logic
        if match:
            test_cache_folder = f"tokenization_{max_length}"
        else:
            test_cache_folder = "tokenization"

        print(f'Using max_length={max_length} for tokenization.')
        print(f'Caching tokenized dataset in folder: {test_cache_folder}')

        def format_and_tokenize(examples):
            texts = [
                f"{tokenizer.cls_token} " + f" {tokenizer.sep_token} ".join(instr_list) + f" {tokenizer.sep_token}"
                for instr_list in examples["instructions"]
            ]
            return tokenizer(texts, truncation=True, max_length=max_length)

        columns_to_remove = [c for c in test_dataset.column_names if c not in ['unique_id']]
        test_cache_tokenization_path = os.path.join(cache_dir, test_cache_folder, f"{args.split}_test.arrow")
        test_dataset = test_dataset.map(format_and_tokenize, batched=True, num_proc=32, remove_columns=columns_to_remove, desc='Tokenizing data for student model...', cache_file_name=test_cache_tokenization_path)

        # Load student model and projector
        projector = None
        if args.checkpoint_dir:
            print(f"--- Loading model from checkpoint directory: {args.checkpoint_dir} ---")
            config = BertConfig(
                vocab_size=len(tokenizer), hidden_size=512, num_attention_heads=8,
                num_hidden_layers=6, intermediate_size=2048, max_position_embeddings=1024
            )
            model = BertForMaskedLM(config=config)
            
            student_weights_path = os.path.join(args.checkpoint_dir, 'student.pth')
            if os.path.exists(student_weights_path):
                model.load_state_dict(torch.load(student_weights_path, weights_only=True, map_location=torch.device('cpu')))
                print(f"--- Loaded student weights from: {student_weights_path} ---")
            else:
                raise FileNotFoundError(f"student.pth not found in checkpoint_dir: {args.checkpoint_dir}")

            projector_weights_path = os.path.join(args.checkpoint_dir, 'projector.pth')
            if os.path.exists(projector_weights_path):
                print(f"--- Found projector.pth, loading projector. ---")
                teacher_dim = teacher_model_info[args.model]["dim"]
                projector = nn.Linear(model.config.hidden_size, teacher_dim)
                projector.load_state_dict(torch.load(projector_weights_path, weights_only=True, map_location=torch.device('cpu')))
                projector.eval()
            else:
                print(f"--- No projector.pth found in checkpoint. ---")
        
        elif args.from_scratch:
            print("--- Initializing STUDENT model from scratch (random weights). ---")
            config = BertConfig(
                vocab_size=len(tokenizer), hidden_size=512, num_attention_heads=8,
                num_hidden_layers=6, intermediate_size=2048, max_position_embeddings=1024
            )
            model = BertForMaskedLM(config=config)
        
        else:
            print("--- Loading default base MLM model. ---")
            model = BertForMaskedLM.from_pretrained(os.path.join(data_dir, f'bert_mlm_{args.split}', 'best_model'))

        # The 'method' argument is now primarily for naming the output file.
        # We need to preserve the init_suffix for correct output naming.
        init_suffix = ""
        if args.from_scratch:
            init_suffix = "_scratch"
        
        print(f"--- Using method '{args.method}' for output naming. ---")
        
        model = model.to(device)
        model.eval()
        if projector is not None:
            projector = projector.to(device)
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    test_unique_ids = test_dataset['unique_id']
    test_dataset = test_dataset.remove_columns(['unique_id'])
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True
    )

    start_time = time.time()
    gpu_monitor = GPU(interval=0.1)
    gpu_monitor.start_measure()

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating embeddings"):
            # --- UNIFIED FORWARD PASS LOGIC ---
            if args.is_teacher:
                # Prepare input based on model type
                if args.model == 'clap':
                    # The default collator batches these into lists of lists
                    keys_batch = batch['keys']
                    instrs_batch = batch['instructions']
                    inputs = [
                        {k: instr for k, instr in zip(keys, instrs)}
                        for keys, instrs in zip(keys_batch, instrs_batch)
                    ]
                else: # For starcoder2, deepseek, etc.
                    instrs_batch = batch["instructions"]
                    inputs = ['\n'.join(i) for i in instrs_batch]
                
                # Forward pass returns numpy array
                numpy_embeddings = model.forward(inputs)
                all_embeddings.append(numpy_embeddings.tolist())
            
            else: # Student model logic
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model.bert(**inputs)
                token_embeddings = outputs.last_hidden_state
                
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled_embeddings = sum_embeddings / sum_mask

                if projector is not None:
                    mean_pooled_embeddings = projector(mean_pooled_embeddings)

                normalized_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)
                all_embeddings.append(normalized_embeddings.cpu().tolist())

    torch.cuda.synchronize() 
    gpu_monitor.stop_measure()
    end_time = time.time()
    print("Inference complete.")
    flat_embeddings = [emb for batch in all_embeddings for emb in batch]
    print("Saving results and metadata...")
    data = {
        "time": end_time - start_time,
        "peak_memory": gpu_monitor.get_memory_usage(peak=True),
        "avg_memory": gpu_monitor.get_memory_usage(average=True),
        "peak_util": gpu_monitor.get_utilization(peak=True) * 100,
        "avg_util": gpu_monitor.get_utilization(average=True) * 100,
    }

    output_path_base = os.path.join(output_dir, "inference", "datasets", args.split, args.model)
    os.makedirs(output_path_base, exist_ok=True)
    
    # Update output filename prefix
    output_filename_prefix = args.model if args.is_teacher else (method + init_suffix)
    metadata_file_path = os.path.join(output_path_base, f"{output_filename_prefix}-metadata.json")

    all_runs_data = []
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, "r") as f:
            existing_data = json.load(f)
            if isinstance(existing_data, dict):
                all_runs_data = existing_data.get("runs", [])
            elif isinstance(existing_data, list):
                all_runs_data = existing_data

    # 2. Append the current run's data
    all_runs_data.append(data)

    # 3. Calculate and print summary statistics
    summary_stats = {}
    if len(all_runs_data) > 0:
        print("\n" + "-"*20 + " Performance Summary " + "-"*20)
        pure_runs = [run for run in all_runs_data if 'mean_time' not in run]

        times = [run['time'] for run in pure_runs]
        avg_memories = [run['avg_memory'] for run in pure_runs]
        avg_utils = [run['avg_util'] for run in pure_runs]
        peak_memories = [run['peak_memory'] for run in pure_runs]
        peak_utils = [run['peak_util'] for run in pure_runs]

        mean_time = np.mean(times)
        std_time = np.std(times) if len(times) > 1 else 0
        mean_avg_mem = np.mean(avg_memories)
        std_avg_mem = np.std(avg_memories) if len(avg_memories) > 1 else 0
        mean_avg_util = np.mean(avg_utils)
        std_avg_util = np.std(avg_utils) if len(avg_utils) > 1 else 0
        mean_peak_mem = np.mean(peak_memories)
        std_peak_mem = np.std(peak_memories) if len(peak_memories) > 1 else 0
        mean_peak_util = np.mean(peak_utils)
        std_peak_util = np.std(peak_utils) if len(peak_utils) > 1 else 0

        print(f"Runs: {len(pure_runs)}")
        print(f"Time (sec):            {mean_time:.2f} ± {std_time:.2f}")
        print(f"Avg Memory (MiB):      {mean_avg_mem:.2f} ± {std_avg_mem:.2f}")
        print(f"Peak Memory (MiB):     {mean_peak_mem:.2f} ± {std_peak_mem:.2f}")
        print(f"Avg GPU Util (%):      {mean_avg_util:.2f} ± {std_avg_util:.2f}")
        print(f"Peak GPU Util (%):     {mean_peak_util:.2f} ± {std_peak_util:.2f}")
        print("-"*59)

        summary_stats = {
            'summary_runs': len(pure_runs),
            'mean_time': mean_time,
            'std_time': std_time,
            'mean_avg_memory': mean_avg_mem,
            'std_avg_memory': std_avg_mem,
            'mean_peak_memory': mean_peak_mem,
            'std_peak_memory': std_peak_mem,
            'mean_avg_util': mean_avg_util,
            'std_avg_util': std_avg_util,
            'mean_peak_util': mean_peak_util,
            'std_peak_util': std_peak_util
        }

    data_to_save = {
        "summary": summary_stats,
        "runs": all_runs_data
    }

    with open(metadata_file_path, "w") as f:
        json.dump(data_to_save, f, indent=4)

    # safe dataset 
    embeddings_save_path = os.path.join(output_path_base, f"{method}{init_suffix}-embeddings")
    print(f"Saving embeddings dataset to {embeddings_save_path}...")
    if not os.path.exists(embeddings_save_path):
        print("Creating final embeddings dataset...")

        def gen():
            for uid, emb in zip(test_unique_ids, flat_embeddings):
                yield {"unique_id": uid, "embedding": emb}

        embeddings_dataset = Dataset.from_generator(gen)
        print(len(embeddings_dataset))
        print(len(test_unique_ids))
        embeddings_dataset.save_to_disk(embeddings_save_path)
        
    print("Done.")