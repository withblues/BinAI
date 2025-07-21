from .base_trainer import BaseTrainer
import torch
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F

class DistillTrainer(BaseTrainer):
    def __init__(
        self,
        bert_model,
        projector, # specific to distillation
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
        projector_save_path="", # specific to distillation
        device='cuda',
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
        self.projector = projector.to(device)
        self.projector_save_path = projector_save_path
        self.criterion = torch.nn.MSELoss()

        # store these for optimizer initialization in base class
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

        # print params
        trainable_params_bert = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_params_proj = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        print(f"Distill Trainer - Trainable BERT Parameters: {trainable_params_bert}")
        print(f"Distill Trainer - Trainable Projector Parameters: {trainable_params_proj}")
        print(f"Distill Trainer - Total Trainable Parameters: {trainable_params_bert + trainable_params_proj}")


    def _initialize_optimizer_and_scheduler(self, lr, betas, weight_decay):
        """Initialize optimizer and scheduler for distillation."""
        self.optim = Adam(
            list(self.model.parameters()) + list(self.projector.parameters()),
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
            anneal_strategy='cos',
            final_div_factor=1e2
        )

    def _save_additional_models(self):
        """Save the projector model."""
        torch.save(self.projector.state_dict(), self.projector_save_path)

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
                # transform input in correct shape
                reshaped_input_for_encode = student_all_parts_batched.view(
                    current_dataloader_batch_size * dynamic_max_len,
                    self.total_seq_len
                )
                
                # forward pass
                encoded_all_parts = self.model.encode(reshaped_input_for_encode)

                # transform back
                encoded_parts_batched_view = encoded_all_parts.view(
                    current_dataloader_batch_size,
                    dynamic_max_len,
                    128
                )

                # mean pooling
                attention_mask_expanded = attention_mask_batched.unsqueeze(-1)
                masked_encoded_parts = encoded_parts_batched_view * attention_mask_expanded
                student_summed_function_embeddings = torch.sum(masked_encoded_parts, dim=1)

                valid_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-5)
                student_mean_function_embeddings = student_summed_function_embeddings / valid_counts

                # normalization and projection layer forward pass
                student_mean_function_embeddings = F.normalize(student_mean_function_embeddings, dim=1)
                projected_student_embeddings = self.projector(student_mean_function_embeddings)
                projected_student_embeddings = F.normalize(projected_student_embeddings, dim=1)

                loss = self.criterion(projected_student_embeddings, teacher_embeddings_batch)

            if train:
                loss_for_total_sum = loss.item() 
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # gradient accumulation
                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(data_loader):
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