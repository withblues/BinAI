import argparse
import pickle
import torch
from utils.data import load_data
import os
from models.tokenizer import AsmTokenizer
from models.dataset import DistillDatasetTruncPadParts
from models.bert import BERT
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm
from torch import nn



class DistillTrainer:
    def __init__(
            self,
            bert_model,
            projector,
            train_dataloader,
            max_len,
            total_seq_len,
            valid_dataloader=None,
            lr= 1e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            log_freq=10,
            num_epochs=20,
            model_save_path="",
            projector_save_path="",
            device='cuda'
    ):
        self.device = device
        self.model = bert_model.to(device)
        self.projector = projector.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # sizes for reshaping
        self.max_len = max_len
        self.total_seq_len = total_seq_len
        
        self.optim = Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.optim_schedul = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=1e-3,
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e2
        )

        self.criterion = torch.nn.MSELoss()
        self.log_freq = log_freq
        self.avg_loss = float('inf')
        self.model_save_path = model_save_path
        self.projector_save_path = projector_save_path
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in self.projector.parameters() if p.requires_grad)

        print("Trainable Parameters:", trainable_params)

    def train(self, epoch):
        _ = self.iteration(epoch, self.train_dataloader)
        avg_loss = self.iteration(epoch, self.valid_dataloader, train=False)

        # save best model
        if avg_loss < self.avg_loss:
            self.avg_loss = avg_loss
            torch.save(self.model.state_dict(), self.model_save_path)
            torch.save(self.projector.state_dict(), self.projector_save_path)

    def iteration(self, epoch, data_loader, train=True):
        mode = "train" if train else "test"
        avg_loss = 0.0

        data_iter = tqdm.tqdm(
            data_loader,
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        if train:
            self.model.train()
            self.projector.train()
        
        else:
            self.model.eval()
            self.projector.eval()

        for i, batch_dict in enumerate(data_iter):
            student_all_parts_batched = batch_dict["student_instruction"].to(self.device)
            parts_attention_mask_batch = batch_dict["student_attention_mask"].to(self.device)
            teacher_embeddings_batch = batch_dict["teacher_embedding"].to(self.device)

            current_dataloader_batch_size = student_all_parts_batched.size(0)

            with torch.set_grad_enabled(train):
                # 1. Reshape student inputs from [batchs_size, max_len, total_seq_len] -> [batch_size * max_len, total_seq_len]
                reshaped_input_for_encode = student_all_parts_batched.view(
                    current_dataloader_batch_size * self.max_len,
                    self.total_seq_len
                )
                
                # 2. Encode all parts
                encoded_all_parts = self.model.encode(reshaped_input_for_encode)

                # 3. Reshape back to [batch_size, max_len, bert_dimension]
                encoded_parts_batched_view = encoded_all_parts.view(
                    current_dataloader_batch_size,
                    self.max_len,
                    128
                )

                # 4. Masked Summation
                attention_mask_expanded = parts_attention_mask_batch.unsqueeze(-1)
                masked_encoded_parts = encoded_parts_batched_view * attention_mask_expanded
                student_summed_function_embeddings = torch.sum(masked_encoded_parts, dim=1)

                # 5. Project teacher embeddings and calculate loss
                projected_teacher_embeddings = self.projector(teacher_embeddings_batch)
                loss = self.criterion(student_summed_function_embeddings, projected_teacher_embeddings)

            if train:
                # 3. backward and optimization only in train
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.optim_schedule.step()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
			avg_loss={avg_loss / len(data_iter)}"
        )
        return avg_loss
    

def simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len):
    # does the same as BERTDataset but removes masking and stacks all combines
    # everything so that we have on instruction for each function
    student_inputs = []

    for t1_char_str, t2_char_str in data_pairs:

        # tokenizer assembly code
        t1_tokens = tokenizer.encode(t1_char_str)
        t2_tokens = tokenizer.encode(t2_char_str)

        # skip masking
        t1_padded = t1_tokens[:seq_len] + [tokenizer.vocab['[PAD]']] * (seq_len - len(t1_tokens))
        t2_padded = t2_tokens[:seq_len] + [tokenizer.vocab['[PAD]']] * (seq_len - len(t2_tokens))

        # adding CLS and SEP tokens
        bert_input = [tokenizer.vocab['[CLS]']] + t1_padded + [tokenizer.vocab['[SEP]']] + t2_padded + [tokenizer.vocab['[SEP]']]
        
        student_inputs.append(
            torch.tensor(bert_input, dtype=torch.long)
        )

    return student_inputs
    

def load_assembly_data(data):
    data_pairs = []
    for item in data:
        for i in range(0, len(item)-1, 2):
            data_pairs.append((item[i].strip(), item[i+1].strip()))

    return data_pairs

def prepare_dataset_for_bert(data, tokenizer, max_len, seq_len, total_seq_length):
    processed_data = []

    for idx, (instructions, teacher_embeddings) in data.items():
        data_pairs = load_assembly_data(instructions)

        # tokenizer each pair
        data_pairs_tokenize = simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len)

        # get length of our function
        len_function = len(data_pairs_tokenize)

        # truncate if function is longer than max_len
        final_function = []
        final_attention_mask = []
        if len_function >= max_len:
            final_function = data_pairs_tokenize[:max_len]
            final_attention_mask = [1] * max_len
        
        else: # else we pad so every function has the same length
            final_function = data_pairs_tokenize
            final_attention_mask = [1] * len_function

            # create dummy instructions
            num_padding_needed = max_len - len_function
            padded_tensor = torch.full((total_seq_length,), tokenizer.vocab['[PAD]'], dtype=torch.long)

            # add dummy instructions
            final_function.extend([padded_tensor] * num_padding_needed)
            final_attention_mask.extend([0] * num_padding_needed)

        # add to output
        processed_data.append({
            'function_id': idx,
            'student_instruction': final_function,
            'student_attention_mask': torch.tensor(final_attention_mask, dtype=torch.long),
            'teacher_embedding': torch.from_numpy(teacher_embeddings).float()
        })

    return processed_data

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_len', default=256)
    parser.add_argument('--seq_len', default=16)
    parser.add_argument('--batch_size', default=2)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = args.data_dir
    output_dir = args.output_dir

    # parameters for training
    max_len = args.max_len
    seq_len = args.seq_len
    # seq_len * 2 (data pairs) + [CLS] + [SEQ] + [SEQ]
    total_seq_len = seq_len * 2 + 3
    batch_size = args.batch_size

    # load input data for student model
    train_data = load_data(os.path.join(data_dir, 'baseline-train-indexed.pkl'))
    # valid_data = load_data(os.path.join(data_dir, 'baseline-valid-indexed.pkl'))
    # if not isinstance(train_data, dict) or not isinstance(valid_data, dict):
    #     raise TypeError('Expected both train_data and valid_data to be dicts')
    
    # load precomputed data from teacher model
    train_embedding_data = load_data(os.path.join(data_dir, 'clap-train-embeddings.pkl'))
    #valid_embedding_data = load_data(os.path.join(data_dir, 'clap-valid-embeddings.pkl'))

    # concatinate datasets
    train_combined = {
        key: (train_data[key], train_embedding_data[key])
        for key in train_data
        if key in train_embedding_data
    }

    # valid_combined = {
    #     key: (valid_data[key], valid_embedding_data[key])
    #     for key in valid_data
    # }

    # load tokenizer
    tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, f"baseline-vocab.txt"))
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # prepare instructions for BERT
    train_data = prepare_dataset_for_bert(train_combined, tokenizer, max_len, seq_len, total_seq_len)
    #valid_data = prepare_dataset_for_bert(valid_combined, tokenizer, max_len, seq_len, total_seq_len)

    # dataset
    train_dataset = DistillDatasetTruncPadParts(train_data)
    #valid_dataset = DistillDatasetTruncPadParts(valid_data)

    def collate_fn(batch):
        # need to stack data pairs into one function
        func_ids = [item["function_id"] for item in batch]
        stacked_instruction = torch.stack(
            [torch.stack(item["student_instruction"], dim=0) for item in batch],
            dim=0
        )

        stacked_attention_mask = torch.stack(
            [item["student_attention_mask"] for item in batch],
            dim=0
        )

        stacked_embedding = torch.stack(
            [item["teacher_embedding"] for item in batch],
            dim=0
        )
    
        return {
            'function_id': func_ids,
            'student_instruction': stacked_instruction,
            'student_attention_mask': stacked_attention_mask,
            'teacher_embedding': stacked_embedding
        }


    # create dataloaders
    train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    #valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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
    teacher_d_size = train_embedding_data[0].shape[0]

    # create projector
    projector = nn.Linear(teacher_d_size, 128)

    # create trainer
    epochs = 10
    distil_trainer = DistillTrainer(
        bert_model=bert_model,
        projector=projector,
        train_dataloader=train_dataloder,
        max_len=max_len,
        total_seq_len=total_seq_len,
        #valid_dataloader=valid_dataloader,
        num_epochs=epochs,
        model_save_path=os.path.join(data_dir, f"distil-model"),
        projector_save_path=os.path.join(data_dir, f'projector-layer'),
        device=device,
    )

    # train
    for epoch in tqdm.tqdm(range(epochs)):
        distil_trainer.train(epoch)
