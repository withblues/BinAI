import argparse
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from datasets import load_from_disk
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig
import torch.nn as nn
import torch.nn.functional as F

model_dims = {
    "clap":       768,
    "starcoder2": 4608,
    "deepseek":   4096,
    "qwen":       3584,
    "codellama":  "/home/wang/Data/llms/CodeLlama-7b-hf",
    "nova":       2048,
}

def compute_embeddings(batch, model, projector, tokenizer, device, column_name):
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token

    texts = [
        f"{cls_token} " + f" {sep_token} ".join(instr_list) + f" {sep_token}"
        for instr_list in batch["instructions"]
    ]

    tokenized = tokenizer(
            texts,
            truncation=True,
            padding='longest',
            max_length=128,
            return_tensors='pt'
        )

    with torch.no_grad():
        inputs = tokenized.to(device)
        outputs = model.bert(**inputs)
        token_embeddings = outputs.last_hidden_state

        # pooling
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        mean_pooled_embeddings = projector(mean_pooled_embeddings)
        normalized_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)

    return {
        column_name: normalized_embeddings.cpu().numpy(),
    }

PROJECTS_TO_REMOVE = {"nmap", "unrar", "z3"}

def remove_cpp(batch):
    projects = batch['project']

    return [project not in PROJECTS_TO_REMOVE for project in projects]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", default="/mnt/ambrym2/datasets/distil")
    parser.add_argument("--max_len", default=1024, type=int)
    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--split", default='project', type=str)
    parser.add_argument("--model", default="clap", type=str.lower, choices=["clap", "starcoder2", "deepseek", "qwen", "codellama", "mlm", "nova"])
    parser.add_argument("--from_scratch", action='store_true', help="Initialize the student model with random weights using the default architecture instead of loading pretrained weights.")
    parser.add_argument("--output_dataset_path", type=str, default=None, help="Path to a dataset to add embedding columns to. If not provided, a path is generated automatically based on other arguments.")
    parser.add_argument("--keep_teacher_embedding", action='store_true', help="Do not remove teacher embedding columns from the dataset.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to checkpoint directory containing student.pth and optionally projector.pth.")
    parser.add_argument("--column_name_prefix", type=str, default=None, help="Prefix for the output column name.")
    args = parser.parse_args()

    data_dir = args.data_dir
    
    if args.output_dataset_path:
        output_path = args.output_dataset_path
        print(f"--- Using specified output dataset path: {output_path} ---")
    else:
        # Build a descriptive output directory name
        print("--- No output_dataset_path specified, generating path automatically. ---")
        output_name_parts = ['assembly_x64_1024']
        if args.checkpoint_dir:
            output_name_parts.append(os.path.basename(args.checkpoint_dir))
        elif args.from_scratch:
            output_name_parts.append("random_model")
        else:
            output_name_parts.append("mlm_model")
        output_name_parts.append('pfastreXML')
        output_dir_name = "_".join(output_name_parts)
        output_path = os.path.join(data_dir, output_dir_name)

    if os.path.exists(output_path):
        print(f"--- Updating existing dataset at: {output_path} ---")
        processed_dataset = load_from_disk(output_path)
    else:
        # This is the FIRST run. Load raw data and filter it.
        print(f"--- Creating new dataset at: {output_path} ---")
        raw_dataset_path = os.path.join(data_dir, f'assembly_x64_1024_clap')
        raw_dataset = load_from_disk(raw_dataset_path)
        print("Filtering C++ projects...")
        processed_dataset = raw_dataset.filter(remove_cpp, num_proc=64, batched=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))

    # --- Model Loading ---
    projector = nn.Identity() # Default projector

    if args.checkpoint_dir:
        print(f"--- Loading model from checkpoint directory: {args.checkpoint_dir} ---")
        config = BertConfig(
            vocab_size=len(tokenizer), hidden_size=512, num_attention_heads=8,
            num_hidden_layers=6, intermediate_size=2048, max_position_embeddings=1024
        )
        student_model = BertForMaskedLM(config=config)
        
        student_weights_path = os.path.join(args.checkpoint_dir, 'student.pth')
        if os.path.exists(student_weights_path):
            student_model.load_state_dict(torch.load(student_weights_path, weights_only=True, map_location=torch.device('cpu')))
            print(f"--- Loaded student weights from: {student_weights_path} ---")
        else:
            raise FileNotFoundError(f"student.pth not found in checkpoint_dir: {args.checkpoint_dir}")

        projector_weights_path = os.path.join(args.checkpoint_dir, 'projector.pth')
        if os.path.exists(projector_weights_path):
            print(f"--- Found projector.pth, loading projector. ---")
            projector = nn.Linear(student_model.config.hidden_size, model_dims[args.model])
            projector.load_state_dict(torch.load(projector_weights_path, weights_only=True, map_location=torch.device('cpu')))
        else:
            print(f"--- No projector.pth found in checkpoint. Using nn.Identity(). ---")

    elif args.from_scratch:
        print("--- Initializing student model from scratch (random weights) ---")
        config = BertConfig(
            vocab_size=len(tokenizer), hidden_size=512, num_attention_heads=8,
            num_hidden_layers=6, intermediate_size=2048, max_position_embeddings=1024
        )
        student_model = BertForMaskedLM(config=config)
    
    else:
        model_path = os.path.join(data_dir, f'bert_mlm_{args.split}', 'best_model')
        print(f"--- Loading default base MLM model from: {model_path} ---")
        student_model = BertForMaskedLM.from_pretrained(model_path)

    student_model = student_model.to(device)
    student_model.eval()
    projector = projector.to(device)
    projector.eval()

    # --- Column Naming ---
    if args.column_name_prefix:
        final_column_name = f"{args.column_name_prefix}_{args.split}_embedding"
    else:
        # Fallback to a generated name if no prefix is given
        print("--- No column_name_prefix specified, generating name automatically. ---")
        if args.checkpoint_dir:
            prefix = os.path.basename(os.path.normpath(args.checkpoint_dir))
            final_column_name = f"{prefix}_{args.split}_embedding"
        elif args.from_scratch:
            final_column_name = f"random_model_{args.split}_embedding"
        else:
            final_column_name = f"mlm_model_{args.split}_embedding"
    print(f"--- Using output column name: {final_column_name} ---")


    cache_dir = os.path.join(data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = f"{final_column_name.replace('/', '_')}_{args.max_len}.arrow"
    cache_file_path = os.path.join(cache_dir, cache_filename)

    actual_cols_to_remove = []
    if not args.keep_teacher_embedding:
        cols_to_remove = ["clap_embedding", f"{args.model}_embedding"]
        actual_cols_to_remove = [
            col for col in cols_to_remove
            if col in processed_dataset.column_names
        ]
        if actual_cols_to_remove:
            print(f"--- Removing teacher embedding columns: {actual_cols_to_remove} ---")
    else:
        print("--- Keeping teacher embedding columns. ---")

    print(f'generate embeddings')
    final_dataset = processed_dataset.map(
        compute_embeddings, 
        batched=True, 
        batch_size=args.batch_size, 
        cache_file_name=cache_file_path, 
        fn_kwargs={
            "model": student_model, 
            "projector": projector,
            "tokenizer": tokenizer,
            "device": device,
            "column_name": final_column_name
            },
        remove_columns=actual_cols_to_remove
        )

    
    final_dataset.save_to_disk(output_path)
    print("precomputing done")