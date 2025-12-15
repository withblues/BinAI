import argparse
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from datasets import load_from_disk
from transformers import BertTokenizerFast, BertForMaskedLM
import torch.nn as nn
import torch.nn.functional as F

model_dims = {
    "clap":       768,
    "starcoder2": 4608,
    "deepseek":   4096,
    "qwen":       3584,
    "codellama":  "/home/wang/Data/llms/CodeLlama-7b-hf",
}

def compute_embeddings(batch, model, projector, tokenizer, device, split):
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
        f'{split}_embedding': normalized_embeddings.cpu().numpy(),
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
    parser.add_argument("--model", default="clap", type=str.lower, choices=["clap", "starcoder2", "deepseek", "qwen", "codellama", "mlm"])
    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = os.path.join(data_dir, f'assembly_x64_1024_{args.model}_pfastreXML')

    if os.path.exists(output_path):
        print(f"--- Updating existing dataset at: {output_path} ---")
        processed_dataset = load_from_disk(output_path)

    else:
        # This is the FIRST run. Load raw data and filter it.
        print(f"--- Creating new dataset at: {output_path} ---")
        #raw_dataset_path = os.path.join(data_dir, f'assembly_x64_1024_{args.model}')
        raw_dataset_path = os.path.join(data_dir, f'assembly_x64_1024_clap')
        raw_dataset = load_from_disk(raw_dataset_path)
        print("Filtering C++ projects...")
        processed_dataset = raw_dataset.filter(remove_cpp, num_proc=64, batched=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))
    student_model = BertForMaskedLM.from_pretrained(os.path.join(data_dir, f'bert_mlm_{args.split}', 'best_model'))

    # load distilled weights
    # print('loading weights')
    # weights_path = os.path.join(data_dir,f'bert_{args.split}', args.model ,'distil_cosine', 'student.pth')
    # student_model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu')))
    # student_model = student_model.to(device)
    # student_model.eval()

    # projector = nn.Linear(student_model.config.hidden_size, model_dims[args.model])
    # weights_path = os.path.join(data_dir, f'bert_{args.split}', args.model, 'distil_cosine', 'projector.pth')
    # projector.load_state_dict(torch.load(weights_path, weights_only=True, map_location=torch.device('cpu')))
    # projector = projector.to(device)
    # projector.eval()

    # for MLM model
    student_model = student_model.to(device)
    student_model.eval()
    projector = nn.Identity()
    projector = projector.to(device)
    projector.eval()

    cache_dir = os.path.join(data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, f"{args.model}_{args.split}_embeddings_{args.max_len}.arrow")

    cols_to_remove = ["clap_embedding", f"{args.model}_embedding"]
    actual_cols_to_remove = [
        col for col in cols_to_remove
        if col in processed_dataset.column_names
    ]
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
            "split": args.split
            },
        remove_columns=actual_cols_to_remove
        )

    
    final_dataset.save_to_disk(output_path)
    print("precomputing done")