import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from datasets import load_from_disk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=30000,       # adjust based on your dataset size
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
    )

    dataset = load_from_disk(args.data_dir)

    def preprocess(example):
        return {
            'text': ' '.join(example['instructions'])
        }
    
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=16)

    tokenizer.train_from_iterator(dataset["text"], trainer)

    tokenizer.save(f'{args.output_dir}/tokenizer.json')