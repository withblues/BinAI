import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer
import json
import time
from src.utils.gpu_stats import GPU
from datasets import load_from_disk, Dataset
from transformers import BertTokenizerFast, BertForMaskedLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from datasets import Features, Value, Sequence


model_dims = {
    "clap":       768,
    "starcoder2": 4608,
    "deepseek":   4096,
    "qwen":       3584,
    "codellama":  "/home/wang/Data/llms/CodeLlama-7b-hf",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--method", default='mse_distil')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model", default='clap', type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    cache_dir = os.path.join(args.data_dir, ".cache", args.model)
    os.makedirs(cache_dir, exist_ok=True)
    method = args.method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"inference on {method} model split {args.split}")

    dataset = load_from_disk(os.path.join(data_dir, 'assembly_x64_1024_clap'))
    with open(os.path.join(data_dir, f"cross_{args.split}_split.json")) as f:
        indices = json.load(f)

    test_ids = set(indices['test'])

    test_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{args.split}_test.arrow")
    test_dataset = dataset.filter(lambda batch: [uid in test_ids for uid in batch["unique_id"]], batched=True, num_proc=16, cache_file_name=test_cache_filter_path)

    if method == 'clap':
        model_path = "hustcw/clap-asm"
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        def tokenize(examples):
            combined_inputs = [
                {k: instr for k, instr in zip(keys, instrs)}
                for keys, instrs in zip(examples["keys"], examples["instructions"])
            ]
            tokenized = tokenizer(
                combined_inputs,
                padding=False,
                #return_tensors="pt",
            )
            return {
                "unique_id": examples["unique_id"],
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "token_type_ids": tokenized["token_type_ids"]
            }
        
        output_features = Features({
            'unique_id': Value('int64'),
            'input_ids': Sequence(Value('int32')),
            'attention_mask': Sequence(Value('int8')),
            'token_type_ids': Sequence(Value('int64'))
        })
                
        columns_to_remove = [c for c in test_dataset.column_names if c not in ['unique_id']]
        test_cache_tokenization_path = os.path.join(cache_dir, 'tokenization', f"{args.split}_clap_test.arrow")
        test_dataset = test_dataset.map(tokenize, batched=True, num_proc=32, remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=test_cache_tokenization_path, features=output_features)

        # model
        clap_model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        ).to(device)
        clap_model.eval()

    else:
        ### tokenizing
        # load custom tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))

        # handle 1024 edge case
        if "1024" in method:
            max_length = 1024
            test_cache_folder = 'tokenization_1024'
        else:
            max_length = 128
            test_cache_folder = 'tokenization'

        # postprocess dataset
        def format_and_tokenize(examples):
            sep_token = tokenizer.sep_token
            cls_token = tokenizer.cls_token
    
            texts = [
                f"{cls_token} " + f" {sep_token} ".join(instr_list) + f" {sep_token}"
                for instr_list in examples["instructions"]
            ]

            # Let tokenizer add CLS at the start and SEP at the end
            tokenized = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
            )
            
            return {
                "unique_id": examples["unique_id"],
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        
        columns_to_remove = [c for c in test_dataset.column_names if c not in ['unique_id']]
        test_cache_tokenization_path = os.path.join(cache_dir, test_cache_folder, f"{args.split}_test.arrow")
        test_dataset = test_dataset.map(format_and_tokenize, batched=True, num_proc=32, remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=test_cache_tokenization_path)

        ### model
        # load pretrained model
        student_model = BertForMaskedLM.from_pretrained(os.path.join(data_dir, f'bert_mlm_{args.split}', 'best_model'))

        if method != 'base':
            # load fine tuned model
            weights_path = os.path.join(data_dir,f'bert_{args.split}', args.model , method, 'student.pth')
            student_model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu')))
            student_model = student_model.to(device)
            

            if 'distil' in method:
                projector = nn.Linear(student_model.config.hidden_size, model_dims[args.model])
                weights_path = os.path.join(data_dir, f'bert_{args.split}', args.model , method, 'projector.pth')
                projector.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu')))
                projector = projector.to(device)
                projector.eval()

        else:
            print('loaded base only')

        student_model = student_model.to(device)
        student_model.eval()

    test_unique_ids = test_dataset['unique_id']
    test_dataset = test_dataset.remove_columns(['unique_id'])
         

    # dataloader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
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


    ### forward pass
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating embeddings"):
            # move to gpu
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            # forward pass
            if method == 'clap':
                normalized_embeddings = clap_model(**inputs)

            else:
                outputs = student_model.bert(**inputs)
                token_embeddings = outputs.last_hidden_state
                
                # mean pooling
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled_embeddings = sum_embeddings / sum_mask

                if 'distil' in method:
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

    output_dir = os.path.join(output_dir, "inference", "datasets", args.split, args.model)
    os.makedirs(output_dir, exist_ok=True)
    metadata_file_path = os.path.join(output_dir, f"{method}-metadata.json")

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
    embeddings_save_path = os.path.join(output_dir, f"{method}-embeddings")
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