import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, BertTokenizerFast
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_dir, 'tokenizer'), exist_ok=True)

    # Create tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Trainer
    trainer = trainers.WordPieceTrainer(
        vocab_size=33555,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
    )

    # Load dataset and preprocess
    dataset = load_from_disk(os.path.join(args.data_dir, 'assembly_x64'))
    def preprocess(example):
        return {'text': ' '.join(example['instructions'])}
    
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names, num_proc=64)

    # Train tokenizer
    tokenizer.train_from_iterator(dataset["text"], trainer)
    
    # Save raw tokenizer
    tokenizer.save(f'{args.output_dir}/tokenizer/tokenizer.json')

    # Wrap in Hugging Face tokenizer
    hf_tokenizer = BertTokenizerFast(
        tokenizer_file=f'{args.output_dir}/tokenizer/tokenizer.json',
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    hf_tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))
