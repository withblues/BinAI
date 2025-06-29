import argparse
import torch
from utils.data import load_data
import os
from models.tokenizer import AsmTokenizer
from models.bert import BERT
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm
from torch import nn
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from models.dataset import CombinedDataset
from datasets import load_from_disk
import torch.nn.functional as F
from utils.gpu_stats import GPU
from torch.utils.data import Dataset, Subset


class DistillTrainer:
    def __init__(
            self,
            bert_model,
            projector,
            train_dataloader,
            total_seq_len,
            gradient_accumulation_steps,
            writer,
            valid_dataloader=None,
            lr= 1e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            log_freq=1,
            num_epochs=20,
            patience=5,
            model_save_path="",
            projector_save_path="",
            device='cuda',
    ):
        self.device = device
        self.model = bert_model.to(device)
        self.projector = projector.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # hyperparameters
        self.total_seq_len = total_seq_len
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.criterion = torch.nn.MSELoss()
        
        # optimizer
        self.optim = Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=1e-3,
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e2
        )

        # early stopping
        self.patience = patience
        self.avg_loss = float('inf')
        self.epochs_no_improve = 0    
        self.early_stop = False

        # logging
        self.writer = writer
        self.log_freq = log_freq

        # model pahts
        self.model_save_path = model_save_path
        self.projector_save_path = projector_save_path

        # print params
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        print("Trainable Parameters:", trainable_params)

    def train(self, epoch):
        train_loss = self.iteration(epoch, self.train_dataloader)
        avg_loss = self.iteration(epoch, self.valid_dataloader, train=False)

        # write to tensorboard
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", avg_loss, epoch)

        # save best model & early stopping
        if avg_loss < self.avg_loss:
            self.avg_loss = avg_loss
            self.epochs_no_improve = 0

            # save model
            torch.save(self.model.state_dict(), self.model_save_path)
            torch.save(self.projector.state_dict(), self.projector_save_path)

        # early stopping
        else:
            self.epochs_no_improve +=1

            if self.epochs_no_improve > self.patience:
                self.early_stop = True

    def iteration(self, epoch, data_loader, train=True):
        mode = "train" if train else "test"
        avg_loss = 0.0

        data_iter = tqdm.tqdm(
            data_loader,
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            #bar_format="{l_bar}{r_bar}"
        )

        if train:
            self.model.train()
            self.projector.train()
            self.optim.zero_grad()
        
        else:
            self.model.eval()
            self.projector.eval()
        

        for i, batch_dict in enumerate(data_iter):
            student_all_parts_batched = batch_dict["student_instruction"].to(self.device)
            attention_mask_batched = batch_dict["student_attention_mask"].to(self.device)
            teacher_embeddings_batch = batch_dict["teacher_embedding"].to(self.device)

            current_dataloader_batch_size, dynamic_max_len, _ = student_all_parts_batched.shape
            with torch.set_grad_enabled(train):
                # reshape student inputs from [batchs_size, max_len, total_seq_len] -> [batch_size * max_len, total_seq_len]

                reshaped_input_for_encode = student_all_parts_batched.view(
                    current_dataloader_batch_size * dynamic_max_len,
                    self.total_seq_len
                )
                
                # encode all parts
                encoded_all_parts = self.model.encode(reshaped_input_for_encode)

                # reshape back to [batch_size, max_len, bert_dimension]
                encoded_parts_batched_view = encoded_all_parts.view(
                    current_dataloader_batch_size,
                    dynamic_max_len,
                    128
                )

                # masked summation so we discard padded instructions
                attention_mask_expanded = attention_mask_batched.unsqueeze(-1)
                masked_encoded_parts = encoded_parts_batched_view * attention_mask_expanded
                student_summed_function_embeddings = torch.sum(masked_encoded_parts, dim=1)

                # mean pooling
                valid_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-5)
                student_mean_function_embeddings = student_summed_function_embeddings / valid_counts

                # normalize student embeddings
                student_mean_function_embeddings = F.normalize(student_mean_function_embeddings, dim=1)

                # project teacher embeddings and calculate loss
                projected_teacher_embeddings = self.projector(teacher_embeddings_batch)
                loss = self.criterion(student_mean_function_embeddings, projected_teacher_embeddings)

            if train:
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # backward and optimization only in train and if accumulation steps
                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(data_loader):
                    self.optim.step()
                    self.optim_schedule.step()
                    self.optim.zero_grad()

                loss_to_display = loss.item() * self.gradient_accumulation_steps
                avg_loss += loss_to_display
            
            else:
                loss_to_display = loss.item()
                avg_loss += loss_to_display
            
            
            data_iter.set_postfix({"loss": f"{loss_to_display:.4f}"})


        print(f"\nEP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}")
        return avg_loss
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seq_len', default=16)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--gradient_accumulation_steps', default=4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = args.data_dir
    output_dir = args.output_dir

    # parameters for training
    seq_len = args.seq_len
    # seq_len * 2 (data pairs) + [CLS] + [SEQ] + [SEQ]
    total_seq_len = seq_len * 2 + 3
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    # load tokenizer
    tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, f"baseline-vocab.txt"))
    PAD_ID = tokenizer.vocab['[PAD]']
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # load teacher embeddings 
    train_teacher_data = load_data(os.path.join(data_dir, 'clap-train-embeddings.pkl'))
    valid_teacher_data = load_data(os.path.join(data_dir, 'clap-valid-embeddings.pkl'))

    # load dataset
    train_dataset = load_from_disk(os.path.join(data_dir, 'distil-train-tokenized'))
    valid_dataset = load_from_disk(os.path.join(data_dir, 'distil-valid-tokenized'))

    # create dataset with teacher mapping
    train_dataset_combined = CombinedDataset(train_dataset, train_teacher_data)
    valid_dataset_combined = CombinedDataset(valid_dataset, valid_teacher_data)
    
    # test TODO
    train_dataset_combined = Subset(train_dataset_combined, range(128))
    valid_dataset_combined = Subset(valid_dataset_combined, range(128))

    def collate_fn(batch, pad_token_id):
        seqs = [sample['student_instruction'] for sample in batch]
        teacher_embeddings = torch.stack([sample['teacher_embedding'] for sample in batch])

        # pad the data 
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=pad_token_id
        )  

        # create attention mask
        attention_masks = (padded_inputs != pad_token_id).any(dim=-1).long()

        return {
            'student_instruction': padded_inputs,
            'student_attention_mask': attention_masks,
            'teacher_embedding': teacher_embeddings
        }


    collate = partial(collate_fn, pad_token_id=PAD_ID)
    train_dataloader = DataLoader(train_dataset_combined, batch_size=batch_size, collate_fn=collate, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset_combined, batch_size=batch_size, collate_fn=collate)

    # load model
    bert_model = BERT(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=2,
        heads=1,
        dropout=0.1,
        device=device
    )
    bert_model.load_state_dict(torch.load(os.path.join(data_dir, f'baseline-model'), map_location=torch.device('cpu')))

    # get d_size of teacher
    teacher_d_size = train_teacher_data[0].shape[0]

    # create projector
    projector = nn.Linear(teacher_d_size, 128)

    # create trainer
    epochs = 2
    writer = SummaryWriter(log_dir=f'{output_dir}/distil-embedding-logs')
    distil_trainer = DistillTrainer(
        bert_model=bert_model,
        projector=projector,
        train_dataloader=train_dataloader,
        gradient_accumulation_steps=gradient_accumulation_steps,
        total_seq_len=total_seq_len,
        writer=writer,
        valid_dataloader=valid_dataloader,
        num_epochs=epochs,
        patience=3,
        model_save_path=os.path.join(output_dir, f"distil-embedding-model.pt"),
        projector_save_path=os.path.join(output_dir, f'distil-projector-layer.pt'),
        device=device,
    )

    # measure GPU Usage
    gpu_monitor = GPU(interval=1.0)
    gpu_monitor.start_measure()

    # train
    for epoch in tqdm.tqdm(range(epochs)):
        distil_trainer.train(epoch)

        if distil_trainer.early_stop:
            print('stop training early')
            break
    
    gpu_monitor.stop_measure()
    print("\n--- GPU Usage Summary ---")
    if gpu_monitor.memory_usage:
        print(f"Peak Memory Usage: {gpu_monitor.get_memory_usage(peak=True):.2f} MB")
        print(f"Average Memory Usage: {gpu_monitor.get_memory_usage(average=True):.2f} MB")
    if gpu_monitor.utilization:
        print(f"Peak Utilization: {gpu_monitor.get_utilization(peak=True)*100:.2f}%")
        print(f"Average Utilization: {gpu_monitor.get_utilization(average=True)*100:.2f}%")
    print("-------------------------")
    writer.close()