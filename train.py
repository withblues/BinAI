import argparse
import torch
import os
from src.models.tokenizer import AsmTokenizer
from src.models.bert import BERT
from src.models.projector import MLPProjector
from torch.utils.data import DataLoader
from torch.nn import Linear
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
from src.models.dataset import CombinedDataset
from src.trainers.distill_trainer import DistillTrainer 
from src.trainers.ranking_trainer import RankingTrainer
from src.utils.data import load_data 
from src.utils.gpu_stats import GPU


def distill_collate_fn(batch, pad_token_id):
    seqs = [sample['student_instruction'] for sample in batch]
    teacher_embeddings = torch.stack([sample['teacher_embedding'] for sample in batch])

    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=pad_token_id
    )  
    attention_masks = (padded_inputs != pad_token_id).any(dim=-1).long()

    return {
        'student_instruction': padded_inputs,
        'student_attention_mask': attention_masks,
        'teacher_embedding': teacher_embeddings
    }

def create_ranking_collate_fn(dataset_tokenized, pad_token_id):
    def ranking_collate_fn(batch):
        anchor_ids = [example["anchor_id"] for example in batch]
        target_ids = [example["target_ids"] for example in batch]
        cosine_scores = [example["cosine_scores"] for example in batch]

        flat_target_ids = [tid for sublist in target_ids for tid in sublist]
        all_ids = anchor_ids + flat_target_ids
        unique_ids = list(set(all_ids))
        
        id_to_inputs = {i: dataset_tokenized[i]["input_ids"] for i in unique_ids}
        id_to_tensor = {i: torch.tensor(seq, dtype=torch.long) for i, seq in id_to_inputs.items()}

        # Use pad_token_id for padding
        padded = torch.nn.utils.rnn.pad_sequence(list(id_to_tensor.values()), batch_first=True, padding_value=pad_token_id)
        attention_mask = (padded.abs().sum(dim=-1) != 0).long()
        
        id_to_index = {id_: idx for idx, id_ in enumerate(id_to_tensor)}

        anchor_indices = [id_to_index[i] for i in anchor_ids]
        target_indices = [[id_to_index[tid] for tid in tlist] for tlist in target_ids]

        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
            "anchor_indices": torch.tensor(anchor_indices),
            "target_indices": torch.tensor(target_indices),
            "cosine_scores": torch.tensor(cosine_scores, dtype=torch.float32),
        }
    return ranking_collate_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument('--mode', type=str, required=True, choices=['distil', 'ranking'],
                        help="Training mode: 'distil' or 'ranking'")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--function_pool', type=str, default='random')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = args.data_dir
    output_dir = args.output_dir

    total_seq_len = args.seq_len * 2 + 3 # seq_len * 2 (data pairs) + [CLS] + [SEQ] + [SEQ]

    # load tokenizer
    tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, f"baseline-vocab.txt"))
    PAD_ID = tokenizer.vocab['[PAD]']
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # load pretrained bert model
    bert_model = BERT(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=2,
        heads=1,
        dropout=0.1,
        device=device
    )
    bert_model.load_state_dict(torch.load(os.path.join(data_dir, f'baseline-model'), map_location=torch.device('cpu')))

    train_dataloader = None
    valid_dataloader = None
    trainer = None
    log_dir = ""
    model_save_path = ""
    projector_save_path = ""

    if args.mode == 'distil':
        print("setting up for Distillation Training...")

        # load teacher embeddings
        train_teacher_data = load_data(os.path.join(data_dir, 'clap/datasets', 'train-embeddings.pkl'))
        valid_teacher_data = load_data(os.path.join(data_dir, 'clap/datasets', 'valid-embeddings.pkl'))

        # load tokenized datasets
        train_dataset_raw = load_from_disk(os.path.join(data_dir, 'distil/datasets', 'train-tokenized'))
        valid_dataset_raw = load_from_disk(os.path.join(data_dir, 'distil/datasets', 'valid-tokenized'))

        # create combined datasets
        train_dataset_combined = CombinedDataset(train_dataset_raw, train_teacher_data)
        valid_dataset_combined = CombinedDataset(valid_dataset_raw, valid_teacher_data)
        
        collate_func = partial(distill_collate_fn, pad_token_id=PAD_ID)
        train_dataloader = DataLoader(train_dataset_combined, batch_size=args.batch_size, collate_fn=collate_func, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset_combined, batch_size=args.batch_size, collate_fn=collate_func, shuffle=False)

        # create projector
        teacher_d_size = train_teacher_data[0].shape[0]
        projector = MLPProjector(128, teacher_d_size)

        log_dir = f'{output_dir}/distil/{args.mode}-logs'
        model_path = os.path.join(output_dir, 'distil/models')
        os.makedirs(model_path, exist_ok=True)
        model_save_path = os.path.join(model_path, f"embedding-bert-model.pt")
        projector_save_path = os.path.join(model_path, f'embedding-projector-layer.pt')

        trainer = DistillTrainer(
            bert_model=bert_model,
            projector=projector,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            total_seq_len=total_seq_len,
            valid_dataloader=valid_dataloader,
            num_epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_save_path=model_save_path,
            projector_save_path=projector_save_path,
            device=device,
            writer=SummaryWriter(log_dir=log_dir)
        )

    elif args.mode == 'ranking':
        print(f"setting up for Ranking Training with dataset fnction-pool-{args.function_pool}...")

        # load datasets
        train_dataset_anchor = load_from_disk(os.path.join(data_dir, 'clap/datasets', f'train-function-pool-{args.function_pool}'))
        train_dataset_tokenized = load_from_disk(os.path.join(data_dir, 'distil/datasets', 'train-tokenized'))
        valid_dataset_anchor = load_from_disk(os.path.join(data_dir, 'clap/datasets', f'valid-function-pool-{args.function_pool}'))
        valid_dataset_tokenized = load_from_disk(os.path.join(data_dir, 'distil/datasets', 'valid-tokenized'))

        # create collate function for ranking mode
        collate_func_train = create_ranking_collate_fn(train_dataset_tokenized, PAD_ID)
        collate_func_valid = create_ranking_collate_fn(valid_dataset_tokenized, PAD_ID)

        train_dataloader = DataLoader(train_dataset_anchor, batch_size=args.batch_size, collate_fn=collate_func_train, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(valid_dataset_anchor, batch_size=args.batch_size, collate_fn=collate_func_valid, shuffle=False, num_workers=4)

        log_dir = f'{output_dir}/distil/ranking-{args.function_pool}-logs'
        model_path = os.path.join(output_dir, 'distil/models')
        os.makedirs(model_path, exist_ok=True)
        model_save_path = os.path.join(model_path, f"ranking-{args.function_pool}-bert-model.pt")

        trainer = RankingTrainer(
            bert_model=bert_model,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            total_seq_len=total_seq_len,
            valid_dataloader=valid_dataloader,
            num_epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_save_path=model_save_path,
            device=device,
            writer=SummaryWriter(log_dir=log_dir)
        )
    else:
        parser.error("Invalid mode selected. Choose 'distillation' or 'ranking'.")

    # start training
    print(f"Starting training in {args.mode} mode...")

    # meassure gpu
    gpu_monitor = GPU(interval=1.0)
    gpu_monitor.start_measure()
    trainer.train()
    
    # show gpu statistics
    gpu_monitor.stop_measure()
    print("\n--- GPU Usage Summary ---")
    if gpu_monitor.memory_usage:
        print(f"Peak Memory Usage: {gpu_monitor.get_memory_usage(peak=True):.2f} MB")
        print(f"Average Memory Usage: {gpu_monitor.get_memory_usage(average=True):.2f} MB")
    if gpu_monitor.utilization:
        print(f"Peak Utilization: {gpu_monitor.get_utilization(peak=True)*100:.2f}%")
        print(f"Average Utilization: {gpu_monitor.get_utilization(average=True)*100:.2f}%")
    print("-------------------------")

    print("training finished.")