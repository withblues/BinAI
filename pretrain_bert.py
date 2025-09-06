import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datasets import load_from_disk
import argparse
import json
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments, BertTokenizerFast, DataCollatorForLanguageModeling
import wandb
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain BERT on project/binary split")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default='project')

    args = parser.parse_args()

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
            'attention_mask': padded_output['attention_mask']
        }

    # tokenize data
    train_dataset = train_dataset.map(format_and_tokenize, batched=True, num_proc=16, remove_columns=train_dataset.column_names, desc='tokenizing data ...')
    val_dataset = val_dataset.map(format_and_tokenize, batched=True, num_proc=16, remove_columns=val_dataset.column_names, desc='tokenizing data ...')

    # collator for MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    ### mmodel
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        intermediate_size=2048,
        max_position_embeddings=1024
    )
    model = BertForMaskedLM(config=config)

    ### training
    # logging
    wandb.init(
            project="assembly_bert",
            name=args.split
        )

    # output dir
    output_dir = os.path.join(output_dir, f"bert_mlm_{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=10000,
        save_steps=10000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        num_train_epochs=6,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        tf32=True,
        report_to='wandb',
        run_name="abert_mlm_pretrain",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # start training
    trainer.train(resume_from_checkpoint=True)

    trainer.save_model(os.path.join(output_dir, "best_model"))
