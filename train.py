import os
from datasets import load_from_disk
import argparse
import json
from transformers import BertForMaskedLM, Trainer, TrainingArguments, BertTokenizerFast, DataCollatorWithPadding, BertConfig, DataCollatorForLanguageModeling
import wandb
from src.models.models import StudentWithProjector, StudentWithCosine, StudentWithInfoNCE, JointAssemblyStudent, StudentWithInBatchCosine, StudentWithInBatchInfoNCE, StudentWithJointInBatch
import torch
from tqdm import tqdm
from src.utils.dataset import CosineDataset, InfoNCEDatasetWithLookup, InBatchInfoNCEDataset
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

class SimpleInBatchCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        all_input_ids = []
        all_attention_masks = []
        all_teacher_embeddings = []
        
        for f in features:
            if len(f["input_ids"]) > 0 and isinstance(f["input_ids"][0], list):
                all_input_ids.extend(f["input_ids"])
                all_attention_masks.extend(f["attention_mask"])
            else:
                all_input_ids.append(f["input_ids"])
                all_attention_masks.append(f["attention_mask"])
                
            if "labels" in f:
                if len(f["input_ids"]) > 0 and isinstance(f["input_ids"][0], list):
                    all_teacher_embeddings.extend(f["labels"])
                else:
                    all_teacher_embeddings.append(f["labels"])
            elif "teacher_embeddings" in f:
                if len(f["input_ids"]) > 0 and isinstance(f["input_ids"][0], list):
                    all_teacher_embeddings.extend(f["teacher_embeddings"])
                else:
                    all_teacher_embeddings.append(f["teacher_embeddings"])
            
        padded = self.tokenizer.pad(
            {"input_ids": all_input_ids, "attention_mask": all_attention_masks},
            padding=True,
            return_tensors="pt"
        )
        
        batch = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }
        
        if all_teacher_embeddings:
            batch["teacher_embeddings"] = torch.tensor(np.array(all_teacher_embeddings), dtype=torch.float32)
            
        return batch

class InBatchInfoNCECollator:
    def __init__(self, tokenizer, mlm_probability=0.0):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        if mlm_probability > 0:
            self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

    def __call__(self, features):
        all_input_ids = []
        all_attention_masks = []
        all_binary_names = []
        all_function_names = []
        anchor_teacher_embeddings = []
        positive_teacher_embeddings = []

        # We first grab all anchors, then all positives
        for feature in features:
            all_input_ids.append(feature["anchor_input_ids"])
            all_attention_masks.append(feature["anchor_attention_mask"])
            all_binary_names.append(feature["anchor_binary_name"])
            all_function_names.append(feature["anchor_function_name"])
            if "anchor_teacher_embedding" in feature:
                anchor_teacher_embeddings.append(feature["anchor_teacher_embedding"])

        for feature in features:
            all_input_ids.append(feature["positive_input_ids"])
            all_attention_masks.append(feature["positive_attention_mask"])
            all_binary_names.append(feature["positive_binary_name"])
            all_function_names.append(feature["positive_function_name"])
            if "positive_teacher_embedding" in feature:
                positive_teacher_embeddings.append(feature["positive_teacher_embedding"])
            
        padded = self.tokenizer.pad(
            {"input_ids": all_input_ids, "attention_mask": all_attention_masks},
            padding=True,
            return_tensors="pt"
        )
        
        batch = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "binary_names": all_binary_names,
            "function_names": all_function_names
        }
        
        if self.mlm_probability > 0:
            clean_input_ids = padded["input_ids"].clone()
            masked_input_ids, mlm_labels = self.mlm_collator.torch_mask_tokens(padded["input_ids"])
            batch["input_ids"] = clean_input_ids
            batch["masked_input_ids"] = masked_input_ids
            batch["mlm_labels"] = mlm_labels
        
        if anchor_teacher_embeddings and positive_teacher_embeddings:
            batch["teacher_embeddings"] = torch.tensor(anchor_teacher_embeddings + positive_teacher_embeddings, dtype=torch.float32)

        return batch

class JointTrainer(Trainer):
    def __init__(self, *args, ol_aux_beta=0.001, ol_aux_horizon=10, use_ol_aux=False, ol_aux_strict_paper=False, nce_start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ol_aux_beta = ol_aux_beta
        self.ol_aux_horizon = ol_aux_horizon
        self.use_ol_aux = use_ol_aux
        self.ol_aux_strict_paper = ol_aux_strict_paper
        self.nce_start_step = nce_start_step
        
        # Accumulators for dot products
        self.dot_mlm_acc = 0.0
        self.dot_distill_acc = 0.0
        self.step_counter = 0

        # Gradient Magnitude Balancers
        self.scale_mlm = 1.0
        self.scale_distill = 1.0


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Safely unwrap the model for Multi-GPU compatibility
        actual_model = self.model.module if hasattr(self.model, "module") else self.model

        if not self.use_ol_aux:
            outputs = model(**inputs, use_ol_aux=self.use_ol_aux)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if self.model.training and isinstance(outputs, dict):
                logs = {f"train/{k}": (v.item() if hasattr(v, 'item') else v) for k, v in outputs.items() if "loss" in k or "w_" in k}
                self.log(logs)
            return (loss, outputs) if return_outputs else loss

        # ------------------------------------------
        # 1. SINGLE FORWARD PASS 
        # ------------------------------------------
        outputs = model(**inputs, use_ol_aux=True) 
        L_main = outputs["nce_loss"]
        L_mlm = outputs["mlm_loss"]
        L_distill = outputs["distill_loss"]

        # Curriculum: disable InfoNCE before nce_start_step
        nce_active = (self.step_counter >= self.nce_start_step)

        # ------------------------------------------
        # 2. ISOLATED PROXY GRADIENTS
        # ------------------------------------------
        if self.model.training:
            self.step_counter += 1
            
            sampled_dot_mlm = 0.0
            sampled_dot_distill = 0.0

            # OL-AUX only runs when InfoNCE is active (no main task = no reference gradient)
            if nce_active and (self.ol_aux_strict_paper or (self.step_counter % self.ol_aux_horizon == 0)):
                proxy_params = []
                if hasattr(actual_model.student.bert.encoder, 'layer'):
                    proxy_params.extend(list(actual_model.student.bert.encoder.layer[-1].parameters()))
                if actual_model.projector is not None:
                    proxy_params.extend(list(actual_model.projector.parameters()))
                    
                proxy_params = [p for p in proxy_params if p.requires_grad]
                
                def get_normalized_proxy_grad(loss_val):
                    grads = torch.autograd.grad(
                        outputs=loss_val, 
                        inputs=proxy_params, 
                        retain_graph=True, 
                        allow_unused=True
                    )
                    flat_grads = []
                    for g, p in zip(grads, proxy_params):
                        if g is not None:
                            flat_grads.append(torch.nan_to_num(g.contiguous().view(-1)))
                        else:
                            flat_grads.append(torch.zeros_like(p.contiguous().view(-1)))
                    
                    combined_grad = torch.cat(flat_grads)
                    norm = torch.norm(combined_grad)
                    if norm > 0:
                        return combined_grad / norm, norm
                    return combined_grad, norm

                grad_main, norm_main = get_normalized_proxy_grad(L_main)
                
                if grad_main is not None:
                    if L_mlm > 0:
                        grad_mlm, norm_mlm = get_normalized_proxy_grad(L_mlm)
                        if grad_mlm is not None:
                            sampled_dot_mlm = torch.nan_to_num(torch.dot(grad_main, grad_mlm)).item()
                            if norm_mlm > 0:
                                self.scale_mlm = torch.clamp(norm_main / norm_mlm, min=0.001, max=1000.0).item()
                    
                    if L_distill > 0:
                        grad_distill, norm_distill = get_normalized_proxy_grad(L_distill)
                        if grad_distill is not None:
                            sampled_dot_distill = torch.nan_to_num(torch.dot(grad_main, grad_distill)).item()
                            if norm_distill > 0:
                                self.scale_distill = torch.clamp(norm_main / norm_distill, min=0.001, max=1000.0).item()

            if nce_active and self.ol_aux_strict_paper:
                self.dot_mlm_acc += sampled_dot_mlm
                self.dot_distill_acc += sampled_dot_distill

            # ------------------------------------------
            # 3. DYNAMIC WEIGHT UPDATE
            # ------------------------------------------
            if nce_active and (self.step_counter % self.ol_aux_horizon == 0):
                with torch.no_grad():
                    if self.ol_aux_strict_paper:
                        new_w_mlm = actual_model.w_mlm + (self.ol_aux_beta * self.dot_mlm_acc)
                        new_w_distill = actual_model.w_distill + (self.ol_aux_beta * self.dot_distill_acc)
                        self.dot_mlm_acc = 0.0
                        self.dot_distill_acc = 0.0
                    else:
                        new_w_mlm = actual_model.w_mlm + (self.ol_aux_beta * sampled_dot_mlm)
                        new_w_distill = actual_model.w_distill + (self.ol_aux_beta * sampled_dot_distill)
                    
                    actual_model.w_mlm.copy_(torch.clamp(new_w_mlm, min=0.01, max=5.0))
                    actual_model.w_distill.copy_(torch.clamp(new_w_distill, min=0.01, max=5.0))

        # ------------------------------------------
        # 4. FINAL COMPUTATION GRAPH & LOGGING
        # ------------------------------------------
        # GradNorm + OL-AUX: Explicit gradient magnitude balancing!
        if nce_active:
            total_loss = (actual_model.lambda_nce * L_main) + \
                         (actual_model.w_mlm * self.scale_mlm * L_mlm) + \
                         (actual_model.w_distill * self.scale_distill * L_distill)
        else:
            # Curriculum phase: only MLM + Distillation
            total_loss = L_mlm + L_distill
        
        outputs["loss"] = total_loss
        outputs["w_mlm"] = actual_model.w_mlm
        outputs["w_distill"] = actual_model.w_distill
        outputs["scale_mlm"] = torch.tensor(self.scale_mlm, device=total_loss.device)
        outputs["scale_distill"] = torch.tensor(self.scale_distill, device=total_loss.device)
        outputs["temperature"] = torch.tensor(actual_model.temperature, device=total_loss.device)
        outputs["nce_active"] = torch.tensor(1.0 if nce_active else 0.0, device=total_loss.device)

        # Logging happens AFTER outputs are correctly populated
        if self.model.training:
            logs = {f"train/{k}": v.item() for k, v in outputs.items() if "loss" in k or "w_" in k or "scale_" in k or k in ("temperature", "nce_active")}
            self.log(logs)
        else:
            logs = {f"eval/{k}": v.item() for k, v in outputs.items() if "loss" in k}
            self.log(logs)
            
        return (total_loss, outputs) if return_outputs else total_loss

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
    parser.add_argument("--filter_truncated", action='store_true', help="Filter out any data that is equal to or exceeds max_len tokens.")
    parser.add_argument("--resume_from_checkpoint", action='store_true', help="Resume training from the last checkpoint in the output directory.")
    parser.add_argument("--batch_size", type=int, default=128, help="Per-device train and eval batch size.")
    
    # Joint training arguments
    parser.add_argument("--lambda_mlm", type=float, default=1.0)
    parser.add_argument("--use_cross_gpu_negatives", action='store_true', help="Use GatherLayer to fetch negatives from all GPUs for InfoNCE scaling.")
    parser.add_argument("--lambda_nce", type=float, default=1.0)
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--top_k", type=int, default=10, help="Number of targets per anchor for InfoNCE/Joint")
    parser.add_argument("--distill_loss_type", type=str, default='mse', choices=['mse', 'cosine', 'kl', 'kl_retrieval', 'topk_kl', 'topk_kl_retrieval', 'pairwiserank', 'pairwiserank_retrieval'], help="Loss function for similarity distillation in joint training.")
    
    # --- New option for OL-AUX dynamic weighting ---
    parser.add_argument("--use_ol_aux", action='store_true', help="Use Online Learning for Auxiliary tasks (OL-AUX) to dynamically weight MLM and Distillation.")
    

    
    parser.add_argument("--temperature_init", type=float, default=0.07, help="Initial temperature for InfoNCE loss (if no scheduler is used, stays constant).")
    parser.add_argument("--distill_temperature", type=float, default=2.0, help="Temperature for KL distillation. Typically much higher than InfoNCE (e.g. 2.0).")
    parser.add_argument("--distill_topk", type=int, default=32, help="Number of top candidates for rank distillation.")
    parser.add_argument("--ol_aux_beta", type=float, default=0.001, help="Learning rate for auxiliary task weights (w).")
    parser.add_argument("--ol_aux_horizon", type=int, default=10, help="Horizon (N) for gradient dot product accumulation.")
    parser.add_argument("--ol_aux_strict_paper", action='store_true', help="Accumulate gradients every step as per the NeurIPS 2019 paper (slower but theoretically sound).")
    parser.add_argument("--nce_start_step", type=int, default=0, help="Step at which to enable InfoNCE loss (curriculum learning). Before this step, only MLM+Distill train.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, set total number of training steps to perform. Overrides num_train_epochs.")

    args = parser.parse_args()
    # print all args
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

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

    if args.filter_truncated:
        print(f'Len of train dataset before filtering: {len(train_dataset)}')
        print(f'Len of val dataset before filtering: {len(val_dataset)}')
        train_dataset = train_dataset.filter(lambda batch: [len(ids) < args.max_len for ids in batch["input_ids"]], batched=True, num_proc=16, desc="filtering truncated train data")
        val_dataset = val_dataset.filter(lambda batch: [len(ids) < args.max_len for ids in batch["input_ids"]], batched=True, num_proc=16, desc="filtering truncated val data")
        print(f'Len of train dataset after filtering: {len(train_dataset)}')
        print(f'Len of val dataset after filtering: {len(val_dataset)}')

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
        if method == 'cosine_in_batch':
            model = StudentWithInBatchCosine(student_model, projector=projector_to_use, distill_loss_type=args.distill_loss_type, temperature=args.temperature_init, distill_temperature=args.distill_temperature, distill_topk=args.distill_topk)
            custom_collate = SimpleInBatchCollator(tokenizer)
            technique = 'cosine_in_batch'
        elif method == 'ft_in_batch':
            # We NEED binary_name and function_name for collision masking!
            technique = 'ft_in_batch'
            sampling = 'random' # uses the offline cosine_random_ft dataset
        elif method == 'joint_in_batch':
            technique = 'joint_in_batch'
            sampling = 'random'
        else:
            split = method.split('_')
            if len(split) > 1:
                sampling = split[-1]
                technique = split[0]
            else:
                technique = method
                sampling = 'random'
                
            # drop unecessary columns
            cols_to_remove = [c for c in ['function_names', 'function_name', 'binary_name'] if c in train_dataset.column_names]
            if cols_to_remove:
                train_dataset = train_dataset.remove_columns(cols_to_remove)
            cols_to_remove_val = [c for c in ['function_names', 'function_name', 'binary_name'] if c in val_dataset.column_names]
            if cols_to_remove_val:
                val_dataset = val_dataset.remove_columns(cols_to_remove_val)
            
        
        if technique == 'cosine':
            ### model
            model = StudentWithCosine(student_model, projector=projector_to_use)
            dataset_name = f'cosine_{sampling}_kd'

        elif technique == 'ft':
            model = StudentWithInfoNCE(student_model, args.top_k, projector=projector_to_use)
            dataset_name = f'cosine_{sampling}_{technique}'
            
        elif technique == 'ft_in_batch':
            model = StudentWithInBatchInfoNCE(student_model, projector=projector_to_use)
            dataset_name = f'cosine_{sampling}_ft'
            
        elif technique == 'joint_in_batch':
            model = StudentWithJointInBatch(student_model, projector=projector_to_use, lambda_nce=args.lambda_nce, lambda_distill=args.lambda_distill, lambda_mlm=args.lambda_mlm, distill_loss_type=args.distill_loss_type, temperature=args.temperature_init, distill_temperature=args.distill_temperature, distill_topk=args.distill_topk)
            dataset_name = f'cosine_{sampling}_ft'
        
        elif technique == 'joint':
            ### model
            model = JointAssemblyStudent(
                student_model, 
                args.top_k, 
                projector=projector_to_use,
                lambda_mlm=args.lambda_mlm,
                lambda_nce=args.lambda_nce,
                lambda_distill=args.lambda_distill,
                distill_loss_type=args.distill_loss_type,
                temperature_init=args.temperature_init
            )
            # Joint training typically uses the KD dataset which has teacher scores
            dataset_name = f'cosine_{sampling}_kd'

        if technique != 'cosine_in_batch':
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

        elif 'ft' in technique or technique == 'joint_in_batch':
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

            if technique == 'ft':
                train_dataset = InfoNCEDatasetWithLookup(train_dataset, train_cosine_lookup, train_id2idx, top_k=args.top_k)
                val_dataset = InfoNCEDatasetWithLookup(val_dataset, val_cosine_lookup, val_id2idx, top_k=args.top_k)
            elif technique == 'ft_in_batch' or technique == 'joint_in_batch':
                train_dataset = InBatchInfoNCEDataset(train_dataset, train_cosine_lookup, train_id2idx)
                val_dataset = InBatchInfoNCEDataset(val_dataset, val_cosine_lookup, val_id2idx)


        if technique != 'joint' and technique != 'cosine_in_batch' and technique != 'ft_in_batch' and technique != 'joint_in_batch':
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
        
        elif technique == 'ft_in_batch':
            custom_collate = InBatchInfoNCECollator(tokenizer)
        elif technique == 'joint_in_batch':
            custom_collate = InBatchInfoNCECollator(tokenizer, mlm_probability=args.mlm_probability)
        


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
    lambda_details = ""
    if technique == 'joint' or technique == 'joint_in_batch':
        if args.use_ol_aux:
            ol_aux_details = f"_olaux_b{args.ol_aux_beta}_h{args.ol_aux_horizon}"
            if args.ol_aux_strict_paper:
                ol_aux_details += "_strict"
        else:
            if args.lambda_mlm != 1.0 or args.lambda_distill != 1.0 or args.lambda_nce != 1.0:
                lambda_details = f"_m{args.lambda_mlm}_d{args.lambda_distill}_n{args.lambda_nce}"
            
    if getattr(args, 'use_cross_gpu_negatives', False):
        lambda_details += "_crossgpu"
            
    nce_details = ""
    if args.nce_start_step > 0:
        nce_details = f"_ncestart{args.nce_start_step}"
        
    distill_type_details = ""
    if args.distill_loss_type != 'mse' and (technique == 'joint' or technique == 'joint_in_batch' or technique == 'cosine_in_batch'):
        distill_type_details = f"_{args.distill_loss_type}"
    filter_trunc_details = "_filter_trunc" if args.filter_truncated else ""
        
    run_name += f'_{args.max_len}' + filter_trunc_details + finetuning_details + init_suffix + ol_aux_details + lambda_details + nce_details + distill_type_details
    print(f'run name: {run_name}')
    wandb.init(
        project=project_name,
        name=run_name,
        config=vars(args) # Log all arguments to wandb config
    )


    # output dir
    output_dir_name = f'{method}_{args.max_len}{finetuning_details}{init_suffix}{ol_aux_details}{lambda_details}{nce_details}{distill_type_details}'
    output_dir = os.path.join(output_dir, f"bert_{args.split}", teacher_type, output_dir_name)
    print(f'output dir: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # Determine label names for evaluation loss calculation
    if technique == 'cosine_in_batch':
        label_names = ['teacher_embeddings']
    elif technique == 'ft_in_batch':
        label_names = ['binary_names']
    elif technique == 'joint_in_batch':
        label_names = ['binary_names', 'teacher_embeddings']
    else:
        label_names = None

    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        #overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=0.2,
        eval_strategy='steps',
        eval_steps=0.20,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=6,
        max_steps=args.max_steps, # 14046 for project split
        logging_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        #tf32=True,
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
        label_names=label_names,
    )

    # Trainer
    if technique == 'joint' or technique == 'joint_in_batch':
        trainer = JointTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=custom_collate,
            use_ol_aux=args.use_ol_aux,
            ol_aux_beta=args.ol_aux_beta,
            ol_aux_horizon=args.ol_aux_horizon,
            ol_aux_strict_paper=args.ol_aux_strict_paper,
            nce_start_step=args.nce_start_step
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=custom_collate,
        )

    # start training
    resume_checkpoint = True if args.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    torch.save(model.student.state_dict(), os.path.join(output_dir, 'student.pth'))

    if hasattr(model, 'projector') and model.projector is not None:
        torch.save(model.projector.state_dict(), os.path.join(output_dir, 'projector.pth'))

    # Save the full wrapper to preserve temperature and OL-AUX weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'joint_wrapper.pth'))

    print('training complete')
