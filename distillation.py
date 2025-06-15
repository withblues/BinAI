import argparse
import torch
from utils.data import load_data, load_json
import os
from models.tokenizer import AsmTokenizer
from models.bert import BERT
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm
from torch import nn
import webdataset as wds
import torch.nn.utils.rnn as rnn_utils
import glob
from functools import partial



class DistillTrainer:
    def __init__(
            self,
            bert_model,
            projector,
            train_dataloader,
            total_seq_len,
            gradient_accumulation_steps,
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

        # hyperparameters
        self.total_seq_len = total_seq_len
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
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

        self.criterion = torch.nn.MSELoss()
        self.log_freq = log_freq
        self.avg_loss = float('inf')
        self.model_save_path = model_save_path
        self.projector_save_path = projector_save_path

        # print params
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

                # project teacher embeddings and calculate loss
                projected_teacher_embeddings = self.projector(teacher_embeddings_batch)
                loss = self.criterion(student_summed_function_embeddings, projected_teacher_embeddings)

            if train:
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # backward and optimization only in train and if accumulation steps
                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(data_loader):
                    self.optim.step()
                    self.optim_schedule.step()
                    self.optim.zero_grad()

            avg_loss += loss.item() * self.gradient_accumulation_steps

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item() * self.gradient_accumulation_steps
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
			avg_loss={avg_loss / len(data_iter)}"
        )
        return avg_loss
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seq_len', default=16)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--gradient_accumulation_steps', default=1)
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

    # teacher mapping
    def map_teacher_data(sample, embedding_dict):
        function_id = int(sample['__key__'])

        if function_id in embedding_dict:
            sample["teacher_embedding"] = torch.from_numpy(embedding_dict[function_id])
            return sample
        
        else:
            return None
        
    # student dataset paths
    shard_dir_train = os.path.join(data_dir, 'distil-train-tokenized-shards')
    shard_dir_valid = os.path.join(data_dir, 'distil-valid-tokenized-shards')

    train_urls = glob.glob(os.path.join(shard_dir_train, 'shards--*.tar.zstd'))
    valid_urls = glob.glob(os.path.join(shard_dir_valid, 'shards--*.tar.zstd'))

    # load lengths of the dataset
    train_metadata = load_json(os.path.join(shard_dir_train, 'metadata.json')) 
    train_length = train_metadata['total_samples']
    valid_metadata = load_json(os.path.join(shard_dir_valid, 'metadata.json')) 
    valid_length = valid_metadata['total_samples']

    # load webdataset
    train_dataset = (
        wds.WebDataset(train_urls, resampled=True)
        .shuffle(1000)
        .decode('torch')
        .map(partial(map_teacher_data, embedding_dict=train_teacher_data))
        .with_length(train_length)
    )


    valid_dataset = (
        wds.WebDataset(valid_urls)
        .decode("torch")
        .map(partial(map_teacher_data, embedding_dict=valid_teacher_data))
        .with_length(valid_length)
    )

    # pad the students to same length
    def pad_collate_fn(batch, pad_token_id):
        # collect data
        student_tensors = [sample['student.pth'].long() for sample in batch]
        teacher_embeddings = [sample['teacher_embedding'] for sample in batch]
        
        # create attention mask
        attention_masks = [torch.ones(t.shape[0], dtype=torch.uint8) for t in student_tensors]

        # pad the student tensors to the max length in this specific batch
        student_instructions_padded = rnn_utils.pad_sequence(
            student_tensors, 
            batch_first=True, 
            padding_value=pad_token_id
        )

        # pad the attention masks to the same length
        student_attention_masks_padded = rnn_utils.pad_sequence(
            attention_masks, 
            batch_first=True, 
            padding_value=0 
        )

        # stack the teacher embeddings
        teacher_embeddings_stacked = torch.stack(teacher_embeddings, dim=0)

        return {
            'student_instruction': student_instructions_padded,
            'student_attention_mask': student_attention_masks_padded,
            'teacher_embedding': teacher_embeddings_stacked
        }

    collate_with_pad = partial(pad_collate_fn, pad_token_id=PAD_ID)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_with_pad)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_with_pad)

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
    epochs = 10
    distil_trainer = DistillTrainer(
        bert_model=bert_model,
        projector=projector,
        train_dataloader=train_dataloader,
        gradient_accumulation_steps=gradient_accumulation_steps,
        total_seq_len=total_seq_len,
        valid_dataloader=valid_dataloader,
        num_epochs=epochs,
        model_save_path=os.path.join(data_dir, f"distil-model"),
        projector_save_path=os.path.join(data_dir, f'projector-layer'),
        device=device,
    )

    # train
    for epoch in tqdm.tqdm(range(epochs)):
        distil_trainer.train(epoch)
