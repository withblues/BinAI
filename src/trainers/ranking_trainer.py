from .base_trainer import BaseTrainer
import torch
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F


class RankingTrainer(BaseTrainer):
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
        num_epochs=20,
        patience=5,
        model_save_path="",
        device="cuda",
    ):
        super().__init__(
            bert_model=bert_model,
            train_dataloader=train_dataloader,
            total_seq_len=total_seq_len,
            gradient_accumulation_steps=gradient_accumulation_steps,
            writer=writer,
            valid_dataloader=valid_dataloader,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            num_epochs=num_epochs,
            patience=patience,
            model_save_path=model_save_path,
            device=device,
        )
        self.criterion = torch.nn.MSELoss()

        # store these for optimizer initialization in base class
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        # Print params
        trainable_params_bert = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Ranking Trainer - Total Trainable Parameters: {trainable_params_bert}")

    def _initialize_optimizer_and_scheduler(self, lr, betas, weight_decay):
        """Initialize optimizer and scheduler for ranking."""
        self.optim = Adam(
            self.model.parameters(),  # Only BERT parameters
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.optim_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=1e-3,
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.num_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=1e2,
        )

    def iteration(self, epoch, data_loader, train=True):
        mode = "train" if train else "eval"
        total_loss = 0.0

        data_iter = tqdm(
            data_loader,
            desc=f"EP_{mode}:{epoch}",
            total=len(data_loader),
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
                # transform input into correct shape
                reshaped_input_for_encode = input_ids.view(
                    n_unique * dynamic_max_len, self.total_seq_len
                )

                # encoder forward pass
                encoded_all_parts = self.model.encode(reshaped_input_for_encode)

                # transform back
                encoded_parts_batched_view = encoded_all_parts.view(
                    n_unique, dynamic_max_len, 128
                )

                # mean pooling
                attention_mask_expanded = attention_mask.unsqueeze(-1)
                function_summed_embeddings = (
                    encoded_parts_batched_view * attention_mask_expanded
                ).sum(dim=1)

                valid_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-5)
                function_mean_embeddings = function_summed_embeddings / valid_counts

                anchor_embeddings = function_mean_embeddings[anchor_indices]
                target_embeddings = function_mean_embeddings[target_indices]

                # normalize
                anchor_norm = F.normalize(anchor_embeddings, dim=-1)
                target_norm = F.normalize(target_embeddings, dim=-1)

                # dot product -> cosine similarity score
                predicted_cosine_similarity_scores = torch.einsum(
                    "bd,bkd->bk", anchor_norm, target_norm
                )

                cosine_scores = cosine_scores.to(self.device)

                loss = self.criterion(predicted_cosine_similarity_scores, cosine_scores)

            if train:
                loss_for_total_sum = loss.item()
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(
                    data_loader
                ):
                    self.optim.step()
                    self.optim_schedule.step()
                    self.optim.zero_grad()

                loss_to_display = loss.item() * self.gradient_accumulation_steps
                total_loss += loss_for_total_sum
            else:
                loss_to_display = loss.item()
                total_loss += loss_to_display

            data_iter.set_postfix({"loss": f"{loss_to_display:.4f}"})

        avg_epoch_loss = total_loss / len(data_iter)
        print(f"\nEP{epoch}, {mode}: avg_loss={avg_epoch_loss:.6f}")
        return avg_epoch_loss
