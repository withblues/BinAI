import argparse
import torch
import os
from models.tokenizer import AsmTokenizer
from models.bert import BERT
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


class RankingTrainer:
    def __init__(
        self,
        bert_model,
        train_dataloader,
        total_seq_len,
        gradient_accumulation_steps,
        writer,
        valid_dataloader=None,
        lr=1e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        log_freq=10,
        num_epochs=20,
        patience=5,
        model_save_path="",
        device="cuda",
    ):
        self.device = device
        self.model = bert_model.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # hyperparameters
        self.total_seq_len = total_seq_len
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.criterion = torch.nn.MSELoss()

        # optimizer
        self.optim = Adam(
            self.model.parameters(),
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
            anneal_strategy="cos",
            final_div_factor=1e2,
        )

        # early stopping
        self.patience = patience
        self.avg_loss = float("inf")
        self.epochs_no_improve = 0
        self.early_stop = False

        # logging
        self.writer = writer
        self.log_freq = log_freq

        # model path
        self.model_save_path = model_save_path

        # print params
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

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

        # early stopping
        else:
            self.epochs_no_improve += 1

            if self.epochs_no_improve > self.patience:
                self.early_stop = True

    def iteration(self, epoch, data_loader, train=True):
        mode = "train" if train else "test"
        avg_loss = 0.0

        data_iter = tqdm.tqdm(
            data_loader,
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        if train:
            self.model.train()
            self.optim.zero_grad()

        else:
            self.model.eval()

        for i, batch_dict in enumerate(data_iter):
            input_ids = batch_dict["input_ids"].to(self.device)
            attention_mask = batch_dict["attention_mask"].to(self.device)
            anchor_indices = batch_dict["anchor_indices"]
            target_indices = batch_dict["target_indices"]
            cosine_scores = batch_dict["cosine_scores"]

            n_unique, dynamic_max_len, _ = input_ids.shape

            with torch.set_grad_enabled(train):
                # reshape input_ids from [n_unique, max_len, total_seq_len] -> [n_unique * max_len, total_seq_len]
                # e.g [704, 128, 35] -> [90 112, 35]
                reshaped_input_for_encode = input_ids.view(
                    n_unique * dynamic_max_len, self.total_seq_len
                )

                # encode all parts
                encoded_all_parts = self.model.encode(reshaped_input_for_encode)
                # reshape back to [n_unique, functions, max_len, bert_dimension]
                encoded_parts_batched_view = encoded_all_parts.view(
                    n_unique, dynamic_max_len, 128
                )

                # masked summation so we discard padded instructions
                attention_mask_expanded = attention_mask.unsqueeze(-1)
                function_summed_embeddings = (
                    encoded_parts_batched_view * attention_mask_expanded
                ).sum(dim=1)

                # count non-pad instructions for mean pooling
                valid_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-5)
                function_mean_embeddings = function_summed_embeddings / valid_counts

                # index anchors and targets
                anchor_embeddings = function_mean_embeddings[anchor_indices]
                target_embeddings = function_mean_embeddings[target_indices]

                # normalize
                anchor_norm = F.normalize(anchor_embeddings, dim=-1)
                target_norm = F.normalize(target_embeddings, dim=-1)

                # calculate cosine similarity
                predicted_cosine_similarity_scores = torch.einsum(
                    "bd,bkd->bk", anchor_norm, target_norm
                )

                # move groundtrugh cosine to gpu and clear gpu memory
                cosine_scores = cosine_scores.to(self.device)

                # loss
                loss = self.criterion(predicted_cosine_similarity_scores, cosine_scores)

            if train:
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # backward and optimization only in train and if accumulation steps
                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(
                    data_loader
                ):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line parameters")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_len", default=16)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--gradient_accumulation_steps", default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    PAD_ID = tokenizer.vocab["[PAD]"]
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # load dataset
    train_dataset_anchor = load_from_disk(
        os.path.join(data_dir, "clap-train-function-pool")
    )
    train_dataset_tokenized = load_from_disk(
        os.path.join(data_dir, "distil-train-tokenized")
    )
    valid_dataset_anchor = load_from_disk(
        os.path.join(data_dir, "clap-valid-function-pool")
    )
    valid_dataset_tokenized = load_from_disk(
        os.path.join(data_dir, "distil-valid-tokenized")
    )

    def create_collate_fn(dataset_tokenized):
        def collate_fn(batch):
            anchor_ids = [example["anchor_id"] for example in batch]
            target_ids = [example["target_ids"] for example in batch]
            cosine_scores = [example["cosine_scores"] for example in batch]

            # get function ids to map
            flat_target_ids = [tid for sublist in target_ids for tid in sublist]
            all_ids = anchor_ids + flat_target_ids
            unique_ids = list(set(all_ids))

            # map function id to input_ids
            id_to_inputs = {i: dataset_tokenized[i]["input_ids"] for i in unique_ids}

            # convert function to tensors
            id_to_tensor = {
                i: torch.tensor(seq, dtype=torch.long)
                for i, seq in id_to_inputs.items()
            }

            # pad to highest len and create attention mask
            padded = pad_sequence(
                list(id_to_tensor.values()), batch_first=True, padding_value=0
            )
            attention_mask = (padded.abs().sum(dim=-1) != 0).long()

            # map back to function ids
            id_to_index = {id_: idx for idx, id_ in enumerate(id_to_tensor)}

            anchor_indices = [id_to_index[i] for i in anchor_ids]
            target_indices = [
                [id_to_index[tid] for tid in tlist] for tlist in target_ids
            ]

            return {
                "input_ids": padded,
                "attention_mask": attention_mask,
                "anchor_indices": torch.tensor(anchor_indices),
                "target_indices": torch.tensor(target_indices),
                "cosine_scores": torch.tensor(cosine_scores, dtype=torch.float32),
            }

        return collate_fn

    # dataloader
    train_dataloader = DataLoader(
        train_dataset_anchor,
        batch_size=batch_size,
        collate_fn=create_collate_fn(train_dataset_tokenized),
        shuffle=True,
        num_workers=4,
    )
    valid_dataloader = DataLoader(
        valid_dataset_anchor,
        batch_size=batch_size,
        collate_fn=create_collate_fn(valid_dataset_tokenized),
        shuffle=False,
        num_workers=4,
    )

    # load model
    bert_model = BERT(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layers=2,
        heads=1,
        dropout=0.1,
        device=device,
    )
    bert_model.load_state_dict(
        torch.load(
            os.path.join(data_dir, f"baseline-model"), map_location=torch.device("cpu")
        )
    )

    # create trainer
    epochs = 2
    writer = SummaryWriter(log_dir=f"{output_dir}/distil-ranking-logs")
    distil_trainer = RankingTrainer(
        bert_model=bert_model,
        train_dataloader=train_dataloader,
        gradient_accumulation_steps=gradient_accumulation_steps,
        total_seq_len=total_seq_len,
        writer=writer,
        valid_dataloader=valid_dataloader,
        num_epochs=epochs,
        patience=3,
        model_save_path=os.path.join(output_dir, f"distil-ranking-model.pt"),
        device=device,
    )

    # train
    for epoch in tqdm.tqdm(range(epochs)):
        distil_trainer.train(epoch)

        if distil_trainer.early_stop:
            print("stop training early")
            break

    writer.close()
