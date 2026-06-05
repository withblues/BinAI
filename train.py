import os
from datasets import load_from_disk
import argparse
import json
from transformers import BertForMaskedLM, Trainer, TrainingArguments, BertTokenizerFast, DataCollatorWithPadding, BertConfig, DataCollatorForLanguageModeling
import wandb
from src.models.models import StudentWithProjector, StudentWithCosine, StudentWithInfoNCE, JointAssemblyStudent
import torch
from tqdm import tqdm
from src.utils.dataset import CosineDataset, InfoNCEDatasetWithLookup
import numpy as np
import random
torch._functorch.config.donated_buffer = False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

model_dims = {
    "clap":       768,
    "starcoder2": 4608,
    "deepseek":   4096,
    "qwen":       3584,
    "nova":      2048,
    "codellama":  "/home/wang/Data/llms/CodeLlama-7b-hf",
}

class JointDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

    def __call__(self, features):
        all_input_ids = []
        all_attention_masks = []
        all_teacher_embeddings = []
        
        for f in features:
            all_input_ids.extend(f["input_ids"])
            all_attention_masks.extend(f["attention_mask"])
            if "teacher_embeddings" in f:
                all_teacher_embeddings.extend(f["teacher_embeddings"])
            
        # Pad sequences
        padded = self.tokenizer.pad(
            {"input_ids": all_input_ids, "attention_mask": all_attention_masks},
            padding=True,
            return_tensors="pt"
        )
        
        # Keep clean input_ids for the second pass
        clean_input_ids = padded["input_ids"].clone()
        
        # Mask tokens for MLM (first pass)
        masked_input_ids, mlm_labels = self.mlm_collator.torch_mask_tokens(padded["input_ids"])
        
        batch = {
            "input_ids": clean_input_ids,       # Used for InfoNCE/Distill
            "masked_input_ids": masked_input_ids, # Used for MLM
            "attention_mask": padded["attention_mask"],
            "mlm_labels": mlm_labels,
        }
        
        if all_teacher_embeddings:
            batch["teacher_embeddings"] = torch.tensor(np.array(all_teacher_embeddings), dtype=torch.float32)
            
        return batch

class JointTrainer(Trainer):
    def __init__(self, *args, ol_aux_beta=0.001, ol_aux_horizon=10, use_ol_aux=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ol_aux_beta = ol_aux_beta
        self.ol_aux_horizon = ol_aux_horizon
        self.use_ol_aux = use_ol_aux
        
        # Accumulators for dot products
        self.dot_mlm_acc = 0.0
        self.dot_distill_acc = 0.0
        self.step_counter = 0


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self.use_ol_aux:
            # Fallback to standard logic if OL-AUX is disabled
            outputs = model(**inputs, use_ol_aux=self.use_ol_aux)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            if self.model.training and isinstance(outputs, dict):
                # Log components and fixed weights
                logs = {f"train/{k}": v.item() for k, v in outputs.items() if "loss" in k or "w_" in k}
                self.log(logs)
                
            return (loss, outputs) if return_outputs else loss
        # ------------------------------------------
        # 1. SINGLE FORWARD PASS (use_ol_aux=True immediately)
        # ------------------------------------------
        outputs = model(**inputs, use_ol_aux=True) 
        L_main = outputs["nce_loss"]
        L_mlm = outputs["mlm_loss"]
        L_distill = outputs["distill_loss"]

        # ------------------------------------------
        # 2. ISOLATED PROXY GRADIENTS (Via Autograd)
        # ------------------------------------------
        if self.model.training:
            self.step_counter += 1
            
            # SAMPLED HORIZON: Only compute proxy gradients on the Nth step
            # This drastically reduces the overhead of OL-AUX, running near baseline speed
            if self.step_counter % self.ol_aux_horizon == 0:
                # We align gradients using the 'Meeting Point': 
                # The last transformer layer and the projector
                proxy_params = []
                if hasattr(model.student.bert.encoder, 'layer'):
                    proxy_params.extend(list(model.student.bert.encoder.layer[-1].parameters()))
                if model.projector is not None:
                    proxy_params.extend(list(model.projector.parameters()))
                    
                proxy_params = [p for p in proxy_params if p.requires_grad]
                eps = 1e-8
                
                # Use autograd.grad instead of .backward()
                # This prevents overwriting p.grad and preserves HF's Gradient Accumulation & AMP
                def get_normalized_proxy_grad(loss_val):
                    grads = torch.autograd.grad(
                        outputs=loss_val, # Removed torch.log to prevent NaN explosion
                        inputs=proxy_params, 
                        retain_graph=True, 
                        allow_unused=True # Crucial: MLM doesn't use the projector
                    )
                    flat_grads = [torch.nan_to_num(g.contiguous().view(-1)) for g in grads if g is not None]
                    if not flat_grads:
                        return None
                    
                    # Combine and normalize to balance gradient magnitudes across tasks 
                    combined_grad = torch.cat(flat_grads)
                    norm = torch.norm(combined_grad)
                    if norm > 0:
                        return combined_grad / norm
                    return combined_grad

                # Calculate dot products silently
                grad_main = get_normalized_proxy_grad(L_main)
                
                sampled_dot_mlm = 0.0
                sampled_dot_distill = 0.0

                if grad_main is not None:
                    if L_mlm > 0:
                        grad_mlm = get_normalized_proxy_grad(L_mlm)
                        if grad_mlm is not None:
                            sampled_dot_mlm = torch.nan_to_num(torch.dot(grad_main, grad_mlm)).item()
                    
                    if L_distill > 0:
                        grad_distill = get_normalized_proxy_grad(L_distill)
                        if grad_distill is not None:
                            sampled_dot_distill = torch.nan_to_num(torch.dot(grad_main, grad_distill)).item()

                # ------------------------------------------
                # 3. DYNAMIC WEIGHT UPDATE
                # ------------------------------------------
                with torch.no_grad():
                    # OUT OF PLACE UPDATE to avoid modifying tensors needed by HF Trainer's backward pass
                    new_w_mlm = model.w_mlm + (self.ol_aux_beta * sampled_dot_mlm)
                    new_w_distill = model.w_distill + (self.ol_aux_beta * sampled_dot_distill)
                    
                    # Clamp to ensure auxiliary tasks don't disappear completely or explode
                    model.w_mlm.copy_(torch.clamp(new_w_mlm, min=0.01, max=5.0))
                    model.w_distill.copy_(torch.clamp(new_w_distill, min=0.01, max=5.0))

            # Log metrics
            logs = {f"train/{k}": v.item() for k, v in outputs.items() if "loss" in k or "w_" in k}
            self.log(logs)
        else:
            # Eval pass
            logs = {f"eval/{k}": v.item() for k, v in outputs.items() if "loss" in k}
            self.log(logs)

        # Return the original total_loss so HF Trainer can do the actual model parameter backward pass
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train BERT on specific objective")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--method", default='distil_mse')
    parser.add_argument("--teacher_type", default='clap')
    parser.add_argument("--max_len", default=128, type=int)
    parser.add_argument("--finetune_checkpoint", type=str, default=None, help="Checkpoint name to start finetuning from (e.g., 'distil_cosine_1024')")
    parser.add_argument("--use_projector_in_ft", action='store_true', help="Use the projector from the checkpoint during finetuning.")
    parser.add_argument("--student_model_name_or_path", type=str, default=None, help="Path or HuggingFace ID of the student model to initialize. If None, defaults to the split specific MLM model.")
    parser.add_argument("--from_scratch", action='store_true', help="Initialize the student model with random weights using the default architecture instead of loading pretrained weights.")
    parser.add_argument("--resume_from_checkpoint", action='store_true', help="Resume training from the last checkpoint in the output directory.")
    
    # Joint training arguments
    parser.add_argument("--lambda_mlm", type=float, default=1.0)
    parser.add_argument("--lambda_nce", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--top_k", type=int, default=10, help="Number of targets per anchor for InfoNCE/Joint")
    parser.add_argument("--distill_loss_type", type=str, default='mse', choices=['mse', 'cosine'], help="Loss function for similarity distillation in joint training.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, set total number of training steps to perform. Overrides num_train_epochs.")

    # OL-AUX arguments
    parser.add_argument("--use_ol_aux", action='store_true', help="Use Online Learning for Auxiliary tasks (OL-AUX) to dynamically weight MLM and Distillation.")
    parser.add_argument("--ol_aux_beta", type=float, default=0.001, help="Learning rate for auxiliary task weights (w).")
    parser.add_argument("--ol_aux_horizon", type=int, default=10, help="Horizon (N) for gradient dot product accumulation.")

    args = parser.parse_args()
    print(f'training on split {args.split} and method {args.method} and teacher {args.teacher_type} and max_len {args.max_len}')
    method = args.method
    # dirs
    data_dir = args.data_dir
    output_dir = args.output_dir
    teacher_type = args.teacher_type

    # handle caching map
    cache_dir = os.path.join(args.data_dir, ".cache", teacher_type, str(args.max_len))
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{args.split}_train.arrow")
    val_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{args.split}_val.arrow")

    ### load dataset
    dataset = load_from_disk(os.path.join(data_dir, f'assembly_x64_1024_{teacher_type}'))
    
    with open(os.path.join(data_dir, f"cross_{args.split}_split.json")) as f:
        indices = json.load(f)

    train_ids = set(indices["train"])
    val_ids = set(indices["val"])

    train_dataset = dataset.filter(lambda batch: [uid in train_ids for uid in batch["unique_id"]], batched=True, num_proc=16, cache_file_name=train_cache_filter_path)
    val_dataset = dataset.filter(lambda batch: [uid in val_ids for uid in batch["unique_id"]], batched=True, num_proc=16, cache_file_name=val_cache_filter_path)


    ### tokenizing
    # load custom tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))

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
            max_length=args.max_len,
        )
        
        return {
            "unique_id": examples["unique_id"],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": examples[f"{teacher_type}_embedding"],
            "function_names": examples['function_name'],
            'binary_name': examples['binary_name'],
        }
        
    # keep clap embedding for now
    columns_to_remove = [c for c in train_dataset.column_names if c not in ['unique_id']]
    
    train_cache_tokenization_path = os.path.join(cache_dir, 'tokenization', f"{args.split}_train.arrow")
    val_cache_tokenization_path = os.path.join(cache_dir, 'tokenization', f"{args.split}_val.arrow")
    train_dataset = train_dataset.map(format_and_tokenize, batched=True, num_proc=os.cpu_count() // 2, remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=train_cache_tokenization_path)
    val_dataset = val_dataset.map(format_and_tokenize, batched=True, num_proc=os.cpu_count() // 2, remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=val_cache_tokenization_path)

    ### model
    if args.from_scratch:
        print("Initializing student model from scratch with default architecture...")
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=6,
            intermediate_size=2048,
            max_position_embeddings=1024
        )
        student_model = BertForMaskedLM(config=config)
    elif args.student_model_name_or_path:
        model_path = args.student_model_name_or_path
        student_model = BertForMaskedLM.from_pretrained(model_path)
    else:
        model_path = os.path.join(data_dir, f'bert_mlm_{args.split}', 'best_model')
        student_model = BertForMaskedLM.from_pretrained(model_path)

    projector_to_use = None 

    if args.finetune_checkpoint:
        checkpoint_dir = os.path.join(args.output_dir, f"bert_{args.split}", args.teacher_type, args.finetune_checkpoint)
        student_weights_path = os.path.join(checkpoint_dir, 'student.pth')
        
        if os.path.exists(student_weights_path):
            print(f"Loading student model from {student_weights_path}")
            student_model.load_state_dict(torch.load(student_weights_path, map_location='cpu'))
            
            if args.use_projector_in_ft:
                projector_weights_path = os.path.join(checkpoint_dir, 'projector.pth')
                if os.path.exists(projector_weights_path):
                    print(f"Found projector weights at {projector_weights_path}")
                    teacher_dim = model_dims[args.teacher_type]
                    projector = torch.nn.Linear(student_model.config.hidden_size, teacher_dim)
                    projector.load_state_dict(torch.load(projector_weights_path, map_location='cpu'))
                    projector_to_use = projector
                else:
                    print("WARNING: --use_projector_in_ft was specified, but no projector.pth found in checkpoint. A new projector will be initialized.")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_dir}. Using default MLM model.")

    if args.use_projector_in_ft:
        if projector_to_use is None:
            print(f"Initializing a new projector for teacher {args.teacher_type} (dim: {model_dims[args.teacher_type]})")
            teacher_dim = model_dims[args.teacher_type]
            projector_to_use = torch.nn.Linear(student_model.config.hidden_size, teacher_dim)
        else:
            print("Using projector loaded from checkpoint.")
    else:
        print("No projector will be used (embedding space will be raw BERT hidden states).")

    if 'distil' in method:
        train_dataset = train_dataset.remove_columns(["function_names", "binary_name", "unique_id"])
        val_dataset = val_dataset.remove_columns(["function_names", "binary_name", "unique_id"])


        #custom_collate = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
        def custom_collate(batch):
            # Standard padding for input_ids / attention_mask
            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]

            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in input_ids],
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(mask) for mask in attention_mask],
                batch_first=True,
                padding_value=0
            )

            # Teacher embeddings: stack safely
            labels = torch.stack([torch.tensor(item['labels'], dtype=torch.float32) for item in batch])
            labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
            labels = torch.nn.functional.normalize(labels, p=2, dim=-1)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        if method == 'distil_cosine':
            model = StudentWithProjector(
                student_model=student_model,
                teacher_dim=model_dims[teacher_type],
                loss_fn='cosine'
            )
        else:
            model = StudentWithProjector(
                student_model=student_model,
                teacher_dim=model_dims[teacher_type],
                loss_fn='mse'
            )
    
    elif 'cosine' in method or 'ft' in method or 'joint' in method:
        split = method.split('_')
        if len(split) > 1:
            sampling = split[-1]
            technique = split[0]
        else:
            technique = method
            sampling = 'random'
            
        # drop unecessary columns
        train_dataset = train_dataset.remove_columns(['function_names', 'binary_name'])
        val_dataset = val_dataset.remove_columns(['function_names', 'binary_name'])
        
        
        if technique == 'cosine':
            ### model
            model = StudentWithCosine(student_model, projector=projector_to_use)
            dataset_name = f'cosine_{sampling}_kd'

        elif technique == 'ft':
            ### dataset
            model = StudentWithInfoNCE(student_model, args.top_k, projector=projector_to_use)
            dataset_name = f'cosine_{sampling}_{technique}'
        
        elif technique == 'joint':
            ### model
            model = JointAssemblyStudent(
                student_model, 
                args.top_k, 
                projector=projector_to_use,
                lambda_mlm=args.lambda_mlm,
                lambda_nce=args.lambda_nce,
                lambda_distill=args.lambda_distill,
                distill_loss_type=args.distill_loss_type
            )
            # Joint training typically uses the KD dataset which has teacher scores
            dataset_name = f'cosine_{sampling}_kd'

        # load cosine dataset
        dataset = load_from_disk(os.path.join(data_dir, f'{dataset_name}', f'cross_{args.split}_split'))
 
        # split data
        train_cache_cosine_path = os.path.join(cache_dir, f'{dataset_name}_filter', f"{args.split}_train.arrow")
        val_cache_cosine_path = os.path.join(cache_dir, f'{dataset_name}_filter', f"{args.split}_val.arrow")
        train_cosine_dataset = dataset.filter(lambda batch: [uid in train_ids for uid in batch["unique_id"]], batched=True, num_proc=32, cache_file_name=train_cache_cosine_path, desc='filter dataset with keys')
        val_cosine_dataset = dataset.filter(lambda batch: [uid in val_ids for uid in batch["unique_id"]], batched=True, num_proc=32, cache_file_name=val_cache_cosine_path, desc='filter dataset with keys')

        if technique == 'cosine' or technique == 'joint':
            ### build lookup table
            # load cosine dataset into ram
            train_cosine_dataset.set_format("numpy", columns=["unique_id", "target_ids", "cosine_scores"])
            val_cosine_dataset.set_format("numpy", columns=["unique_id", "target_ids", "cosine_scores"])

            train_cosine_cols = train_cosine_dataset[:]
            val_cosine_cols = val_cosine_dataset[:]

            train_cosine_lookup = {
                    int(uid): ([int(tid) for tid in targets], scores)
                    for uid, targets, scores in tqdm(
                        zip(train_cosine_cols["unique_id"], train_cosine_cols["target_ids"], train_cosine_cols["cosine_scores"]),
                        total=len(train_cosine_cols["unique_id"]),
                        desc="Building train lookup"
                    )
                }
            
            val_cosine_lookup = {
                    int(uid): ([int(tid) for tid in targets], scores)
                    for uid, targets, scores in tqdm(
                        zip(val_cosine_cols["unique_id"], val_cosine_cols["target_ids"], val_cosine_cols["cosine_scores"]),
                        total=len(val_cosine_cols["unique_id"]),
                        desc="Building val lookup"
                    )
                }
            
            ### build lookup table
            final_train_uids = train_dataset["unique_id"]
            final_val_uids = val_dataset["unique_id"]

            train_id2idx = {uid: i for i, uid in tqdm(enumerate(final_train_uids), total=len(final_train_uids), desc="Building train id2idx")}
            val_id2idx = {uid: i for i, uid in tqdm(enumerate(final_val_uids), total=len(final_val_uids), desc="Building val id2idx")}

            train_dataset = CosineDataset(train_dataset, train_cosine_lookup, train_id2idx, top_k=args.top_k)
            val_dataset = CosineDataset(val_dataset, val_cosine_lookup, val_id2idx, top_k=args.top_k)
            
            if technique == 'joint':
                custom_collate = JointDataCollator(tokenizer, mlm_probability=args.mlm_probability)

        elif technique == 'ft':
            train_cosine_dataset.set_format("numpy", columns=["unique_id", "positive_ids", "negative_ids"])
            val_cosine_dataset.set_format("numpy", columns=["unique_id", "positive_ids", "negative_ids"])

            # load into RAM
            train_cosine_cols = train_cosine_dataset[:]
            val_cosine_cols = val_cosine_dataset[:]

            train_cosine_lookup = {}
            for anchor, positives, negatives in tqdm(
                zip(train_cosine_cols["unique_id"], train_cosine_cols["positive_ids"], train_cosine_cols["negative_ids"]),
                total=len(train_cosine_cols["unique_id"]),
                desc="Building FT train lookup"
            ):
                anchor_int = int(anchor)
                if anchor_int not in train_cosine_lookup:
                    train_cosine_lookup[anchor_int] = []
                
                for positive_id in positives:
                    train_cosine_lookup[anchor_int].append({
                        'positive_id': int(positive_id), 
                        'negative_ids': [int(n) for n in negatives]
                    })
            val_cosine_lookup = {}
            for anchor, positives, negatives in tqdm(
                zip(val_cosine_cols["unique_id"], val_cosine_cols["positive_ids"], val_cosine_cols["negative_ids"]),
                total=len(val_cosine_cols["unique_id"]),
                desc="Building FT val lookup"
            ):
                anchor_int = int(anchor)
                if anchor_int not in val_cosine_lookup:
                    val_cosine_lookup[anchor_int] = []

                for positive_id in positives:
                    val_cosine_lookup[anchor_int].append({
                        'positive_id': int(positive_id), 
                        'negative_ids': [int(n) for n in negatives]
                    })

            ### build lookup table
            final_train_uids = train_dataset["unique_id"]
            final_val_uids = val_dataset["unique_id"]

            train_id2idx = {uid: i for i, uid in tqdm(enumerate(final_train_uids), total=len(final_train_uids), desc="Building train id2idx")}
            val_id2idx = {uid: i for i, uid in tqdm(enumerate(final_val_uids), total=len(final_val_uids), desc="Building val id2idx")}

            train_dataset = InfoNCEDatasetWithLookup(train_dataset, train_cosine_lookup, train_id2idx, top_k=args.top_k)
            val_dataset = InfoNCEDatasetWithLookup(val_dataset, val_cosine_lookup, val_id2idx, top_k=args.top_k)


        if technique != 'joint':
            def custom_collate(features):
                all_input_ids = []
                all_attention_masks = []
                all_labels = []

                # Loop through each example in the batch
                for feature in features:
                    all_input_ids.extend(feature['input_ids'])
                    all_attention_masks.extend(feature['attention_mask'])
                    all_labels.append(feature['labels'])

                # padding
                padded_batch = tokenizer.pad(
                    {"input_ids": all_input_ids, "attention_mask": all_attention_masks},
                    padding='longest',
                    return_tensors='pt',
                )

                labels_np = np.array(all_labels)
                batch_labels = torch.from_numpy(labels_np).float()
                padded_batch['labels'] = batch_labels

                return padded_batch
        


    # training
    #logging
    project_name = f"bert_{teacher_type}_{method}"
    run_name = args.split

    finetuning_details = ""
    if args.finetune_checkpoint:
        finetuning_details += f"_from_{args.finetune_checkpoint}"
        if args.use_projector_in_ft:
            finetuning_details += "_with_proj"
        else:
            finetuning_details += "_no_proj"
    
    # Add initialization details
    init_suffix = ""
    if args.from_scratch:
        init_suffix = "_scratch"
    elif args.student_model_name_or_path:
        init_suffix = "_custom"
    else:
        init_suffix = ""
        
    ol_aux_details = ""
    if technique == 'joint' and args.use_ol_aux:
        ol_aux_details = f"_olaux_b{args.ol_aux_beta}_h{args.ol_aux_horizon}"
        
    run_name += f'_{args.max_len}' + finetuning_details + init_suffix + ol_aux_details
    print(f'run name: {run_name}')
    wandb.init(
        project=project_name,
        name=run_name,
        config=vars(args) # Log all arguments to wandb config
    )


    # output dir
    output_dir_name = f'{method}_{args.max_len}{finetuning_details}{init_suffix}{ol_aux_details}'
    output_dir = os.path.join(output_dir, f"bert_{args.split}", teacher_type, output_dir_name)
    print(f'output dir: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=0.20,
        eval_strategy='steps',
        eval_steps=0.20,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        num_train_epochs=6,
        max_steps=args.max_steps, # 14046 for project split
        logging_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        tf32=True,
        report_to='wandb',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        save_safetensors=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        torch_compile=True,
    )

    # Trainer
    if technique == 'joint':
        trainer = JointTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=custom_collate,
            use_ol_aux=args.use_ol_aux,
            ol_aux_beta=args.ol_aux_beta,
            ol_aux_horizon=args.ol_aux_horizon
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=custom_collate,
        )

    # start training
    resume_checkpoint = True if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    torch.save(model.student.state_dict(), os.path.join(output_dir, 'student.pth'))

    if hasattr(model, 'projector') and model.projector is not None:
        torch.save(model.projector.state_dict(), os.path.join(output_dir, 'projector.pth'))

    print('training complete')
