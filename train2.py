import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_from_disk
import argparse
import json
from transformers import BertForMaskedLM, Trainer, TrainingArguments, BertTokenizerFast, DataCollatorWithPadding
import wandb
from src.models.models import StudentWithProjector, StudentWithCosine, StudentWithInfoNCE
import torch
from tqdm import tqdm
from src.models.dataset import CosineDataset
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT on specific objective")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--method", default='mse_distil')

    args = parser.parse_args()
    print(f'training on split {args.split} and method {args.method}')
    method = args.method

    # dirs
    data_dir = args.data_dir
    output_dir = args.output_dir

    # handle caching map
    cache_dir = os.path.join(args.data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{args.split}_train.arrow")
    val_cache_filter_path = os.path.join(cache_dir, 'dataset_filter', f"{args.split}_val.arrow")

    ### load dataset
    dataset = load_from_disk(os.path.join(data_dir, 'assembly_x64'))
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
            max_length=128,
        )
        
        return {
            "unique_id": examples["unique_id"],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": examples["clap_embedding"],
            "function_names": examples['function_name'],
            'binary_name': examples['binary_name'],
        }
        
    # keep clap embedding for now
    columns_to_remove = [c for c in train_dataset.column_names if c not in ['unique_id']]
    
    train_cache_tokenization_path = os.path.join(cache_dir, 'tokenization', f"{args.split}_train.arrow")
    val_cache_tokenization_path = os.path.join(cache_dir, 'tokenization', f"{args.split}_val.arrow")
    train_dataset = train_dataset.map(format_and_tokenize, batched=True, num_proc=os.cpu_count(), remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=train_cache_tokenization_path)
    val_dataset = val_dataset.map(format_and_tokenize, batched=True, num_proc=os.cpu_count(), remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=val_cache_tokenization_path)

    ### model
    student_model = BertForMaskedLM.from_pretrained(os.path.join(data_dir, f'bert_mlm_{args.split}', 'best_model'))

    if 'distil' in method:
        train_dataset = train_dataset.remove_columns(["function_names", "binary_name", "unique_id"])
        val_dataset = val_dataset.remove_columns(["function_names", "binary_name", "unique_id"])

        custom_collate = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')
        
        if method == 'cosine_distil':
            model = StudentWithProjector(
                student_model=student_model,
                teacher_dim=768,
                loss_fn='cosine'
            )
        else:
            model = StudentWithProjector(
                student_model=student_model,
                teacher_dim=768,
                loss_fn='mse'
            )
    
    elif 'cosine' in method or 'ft' in method:
        split = method.split('_')
        sampling = split[-1]
        technique = split[0]

        ### model
        if technique == 'cosine':
            model = StudentWithCosine(student_model)
            dataset_name = f'cosine_{sampling}'

        elif technique == 'ft':
            model = StudentWithInfoNCE(student_model, 10)
            dataset_name = f'cosine_{sampling}_{technique}'

        ### dataset
        # drop unecessary columns
        train_dataset = train_dataset.remove_columns(["labels", 'function_names', 'binary_name'])
        val_dataset = val_dataset.remove_columns(["labels", 'function_names', 'binary_name'])


        # load cosine dataset
        dataset = load_from_disk(os.path.join(data_dir, f'{dataset_name}', f'cross_{args.split}_split'))

        train_cache_cosine_path = os.path.join(cache_dir, f'{dataset_name}_filter', f"{args.split}_train.arrow")
        val_cache_cosine_path = os.path.join(cache_dir, f'{dataset_name}_filter', f"{args.split}_val.arrow")
        train_cosine_dataset = dataset.filter(lambda batch: [uid in train_ids for uid in batch["unique_id"]], batched=True, num_proc=os.cpu_count(), cache_file_name=train_cache_cosine_path, desc='filter dataset with keys')
        val_cosine_dataset = dataset.filter(lambda batch: [uid in val_ids for uid in batch["unique_id"]], batched=True, num_proc=os.cpu_count(), cache_file_name=val_cache_cosine_path, desc='filter dataset with keys')

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

        train_dataset = CosineDataset(train_dataset, train_cosine_lookup, train_id2idx, technique)
        val_dataset = CosineDataset(val_dataset, val_cosine_lookup, val_id2idx, technique)

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
        


    ## training
    #logging
    wandb.init(
        project=f"bert_{method}",
        name=args.split
    )

    # output dir
    output_dir = os.path.join(output_dir, f"bert_{args.split}", method)
    os.makedirs(output_dir, exist_ok=True)

    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=0.20,
        eval_strategy='steps',
        eval_steps=0.20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=6,
        logging_steps=100,
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
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=custom_collate,
    )

    # start training
    trainer.train()

    torch.save(model.student.state_dict(), os.path.join(output_dir, 'student.pth'))

    if method == 'mse_distil' or method == 'cosine_distil':
        torch.save(model.projector.state_dict(), os.path.join(output_dir, 'projector.pth'))

    print('training complete')
