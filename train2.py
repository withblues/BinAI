import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets import load_from_disk
import argparse
import json
from transformers import BertForMaskedLM, Trainer, TrainingArguments, BertTokenizerFast, DataCollatorForLanguageModeling
import wandb
from src.models.models import StudentWithProjector
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT on specific objective")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')
    parser.add_argument("--method", default='mse_distil')

    args = parser.parse_args()

    method = args.method

    # dirs
    data_dir = args.data_dir
    output_dir = args.output_dir

    ### load dataset
    dataset = load_from_disk(os.path.join(data_dir, 'assembly_x64'))
    with open(os.path.join(data_dir, f"cross_{args.split}_split.json")) as f:
        indices = json.load(f)

    train_dataset = dataset.select(indices["train"])
    val_dataset = dataset.select(indices["val"])

    ### tokenizing
    # load custom tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))

    # postprocess dataset
    def format_and_tokenize(examples):
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        input_ids_list = []
        max_len = 1024

        for instructions_list in examples['instructions']:
            ids = [cls_id]
            for instr in instructions_list:
                ids += tokenizer.encode(instr, add_special_tokens=False)
                ids += [sep_id]
            
            # Truncate the list of token IDs if it exceeds the max length
            # We subtract 1 to account for the final SEP token that will be added.
            if len(ids) > max_len:
                ids = ids[:max_len]
            
            input_ids_list.append(ids)

        # Use the tokenizer's built-in pad method
        padded_output = tokenizer.pad(
            {"input_ids": input_ids_list},
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

        return {
            'input_ids': padded_output['input_ids'],
            'attention_mask': padded_output['attention_mask'],
            'labels': examples['clap_embedding']
        }
    
    # keep clap embedding for now
    columns_to_remove = [c for c in train_dataset.column_names if c not in ['clap_embedding']]
    
    # handle caching map
    cache_dir = os.path.join(args.data_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_path = os.path.join(cache_dir, f"{args.split}.arrow")
    
    train_dataset = train_dataset.map(format_and_tokenize, batched=True, num_proc=16, remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=cache_file_path)
    val_dataset = val_dataset.map(format_and_tokenize, batched=True, num_proc=16, remove_columns=columns_to_remove, desc='tokenizing data ...', cache_file_name=cache_file_path)

    ### model
    student_model = BertForMaskedLM.from_pretrained(os.path.join(data_dir, f'bert_mlm_{args.split}', 'best_model'))

    if 'distil' in method:
        def custom_collate(features):
            batch = {}
            batch['input_ids'] = torch.stack([torch.tensor(f["input_ids"]) for f in features])
            batch['attention_mask'] = torch.stack([torch.tensor(f["attention_mask"]) for f in features])

            batch["labels"] = torch.stack([torch.tensor(f["clap_embedding"]) for f in features])

            return batch
        
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


    ### training
    # logging
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
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=4,
        num_train_epochs=6,
        logging_steps=100,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        tf32=True,
        report_to='wandb',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_safetensors=False
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
    trainer.train(resume_from_checkpoint=True)

    torch.save(model.student.state_dict(), os.path.join(output_dir, 'student.pth'))

    if method == 'mse_distil' or method == 'cosine_distil':
        torch.save(model.projector.state_dict(), os.path.join(output_dir, 'projector.pth'))

    print('training complete')
