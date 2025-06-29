import torch
from tqdm import tqdm


class BaseTrainer:
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
        device='cuda',
    ):
        self.device = device
        self.model = bert_model.to(device) # BERT model is common
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.total_seq_len = total_seq_len
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Early stopping
        self.patience = patience
        self.best_loss = float('inf') # Renamed to best_loss for clarity
        self.epochs_no_improve = 0    
        self.early_stop = False

        # logging
        self.writer = writer
        self.num_epochs = num_epochs # Store for OneCycleLR
        
        # model saving
        self.model_save_path = model_save_path

        # print params (will be overridden/extended in subclasses)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Base Trainer - Trainable BERT Parameters: {trainable_params}")


    def _initialize_optimizer_and_scheduler(self, lr, betas, weight_decay):
        """subclasses will implement this to define their specific optimizer and scheduler."""
        raise NotImplementedError

    def iteration(self, epoch, data_loader, train=True):
        """subclasses will implement their specific forward/backward pass and loss calculation."""
        raise NotImplementedError

    def train(self): # Modified to handle epoch loop internally
        # inialize optimizer
        self._initialize_optimizer_and_scheduler(self.lr, self.betas, self.weight_decay)

        try:
            for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
                train_loss = self.iteration(epoch, self.train_dataloader, train=True)
                val_loss = self.iteration(epoch, self.valid_dataloader, train=False)

                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)

                # Early stopping and saving
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.model_save_path)
                    self._save_additional_models() # subclass for projector

                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve > self.patience:
                        self.early_stop = True
                        print(f"Epoch {epoch}: Early stopping triggered. Validation loss did not improve for {self.patience} epochs.")
                        break

        except KeyboardInterrupt:
            print("training interrupted.")
        finally:
            self.writer.close()

    def _save_additional_models(self):
        """Placeholder for saving specific models (like projector). Subclasses override this."""
        pass

