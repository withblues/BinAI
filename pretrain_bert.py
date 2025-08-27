from datasets import load_from_disk
import argparse
import json
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments, BertTokenizerFast, DataCollatorForLanguageModeling
import wandb
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating Train/Val/Test Split")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    # dirs
    data_dir = args.data_dir
    output_dir = args.output_dir

    ### load dataset
    dataset = load_from_disk(os.path.join(data_dir, 'data'))
    with open(os.path.join(data_dir, "split_indices.json")) as f:
        indices = json.load(f)

    train_dataset = dataset.select(indices["train"])
    val_dataset = dataset.select(indices["val"])
    

    # postprocess dataset
    # def flatten_instructions(example):
    #     return {"text": "; ".join(example["instructions"])}
    
    # train_dataset = train_dataset.map(flatten_instructions, remove_columns=train_dataset.column_names, num_proc=4)
    # val_dataset = val_dataset.map(flatten_instructions, remove_columns=val_dataset.column_names, num_proc=4)

    ### tokenizing
    # load custom tokenizer
    tokenizer = BertTokenizerFast(tokenizer_file="assembly_tokenizer.json")
    # compute token lengths for your assembly dataset
    def compute_length(example):
        tokens = tokenizer(" ; ".join(example["instructions"]))
        return {"length": len(tokens["input_ids"])}

    # map over the dataset in parallel
    dataset_with_lengths = dataset.map(compute_length, num_proc=16)

    # now you can analyze
    lengths = dataset_with_lengths["length"]
    print("Max length:", max(lengths))
    print("Median length:", sorted(lengths)[len(lengths)//2])

    exit()
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

    # tokenize data
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=16)
    val_dataset = val_dataset.map(tokenize, batched=True, num_proc=16)

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
    wandb.init(project="assembly_bert")

    # training args
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "bert_mlm"),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=100,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warumup_steps=0.1,
        report_to='wandb',
        run_name="abert_mlm_pretrain"
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
    trainer.train()