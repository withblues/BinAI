import torch.nn as nn
import torch.nn.functional as F
import torch


    
class StudentWithProjector(nn.Module):
    def __init__(self, student_model, teacher_dim, loss_fn):
        super().__init__()
        self.student = student_model
        self.projector = nn.Linear(self.student.config.hidden_size, teacher_dim)

        self.loss_fn = loss_fn
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'cosine':
            self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # student model forward pass
        outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask

        # project student to teacher
        s_proj = self.projector(mean_embedding)
        s_proj = F.normalize(s_proj, p=2, dim=-1)

        loss = None
        if labels is not None:
            # mse
            if self.loss_fn == 'mse':
                loss = self.criterion(s_proj, labels)
            
            # cosine embedding loss
            elif self.loss_fn == 'cosine':
                target = torch.ones(s_proj.size(0), device=s_proj.device)

                loss = self.criterion(s_proj, labels, target)

        return {
            'loss': loss,
            'logits': s_proj
        }
    
class StudentWithCosine(nn.Module):
    def __init__(self, student_model, projector=None):
        super().__init__()
        self.student = student_model
        self.projector = projector
        self.criterion = nn.MSELoss()


    def forward(self, input_ids, attention_mask=None, labels=None):
        # student model forward pass
        outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        if self.projector:
            projected_embeddings = self.projector(mean_pooled_embeddings)
            normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        else:
            normalized_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)

        # split to anchor and multiple targets
        batch_size = labels.shape[0]
        num_targets = labels.shape[1]

        grouped_embeddings = normalized_embeddings.view(batch_size, num_targets + 1, -1)
        anchor_embeddings = grouped_embeddings[:, 0, :]
        target_embeddings = grouped_embeddings[:, 1:, :]

        # caculate cosine similarity 
        predicted_scores = torch.einsum('bh,bkh->bk', anchor_embeddings, target_embeddings)

        loss = None
        if labels is not None:
            loss = self.criterion(predicted_scores, labels)

        return {
            "loss": loss,
            "logits": predicted_scores
        }


class StudentWithInfoNCE(nn.Module):
    def __init__(self, student_model, top_k, projector=None):
        super().__init__()
        self.student = student_model
        self.criterion = nn.CrossEntropyLoss()
        self.num_targets_per_anchor = top_k
        self.projector = projector
        self._keys_to_ignore_on_save = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        # student model forward pass
        outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask
        
        if self.projector:
            projected_embeddings = self.projector(mean_pooled_embeddings)
            normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        else:
            normalized_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)

        # split to anchor and multiple targets
        num_embeddings_per_group = 1 + self.num_targets_per_anchor
        batch_size = normalized_embeddings.shape[0] // num_embeddings_per_group
        
        grouped_embeddings = normalized_embeddings.view(batch_size, num_embeddings_per_group, -1)
        
        anchor_embeddings = grouped_embeddings[:, 0, :]
        target_embeddings = grouped_embeddings[:, 1:, :]

        # caculate cosine similarity 
        scores = torch.einsum('bh,bkh->bk', anchor_embeddings, target_embeddings)
        
        # calculate InfoNCE Loss
        loss = None
        if labels is not None: # `labels` isn't used, but it's good practice for Trainer
            # The "correct" class is always index 0, because we structured the data
            # with the positive sample first.
            ground_truth_labels = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
            
            loss = self.criterion(scores, ground_truth_labels)

        return {
            "loss": loss,
            "logits": scores
        }
    
class JointAssemblyStudent(nn.Module):
    def __init__(self, student_model, num_targets_per_anchor, projector=None, lambda_mlm=1.0, lambda_nce=1.0, lambda_distill=1.0, distill_loss_type='mse'):
        super().__init__()
        # student_model should be initialized as BertForMaskedLM
        self.student = student_model 
        self.num_targets_per_anchor = num_targets_per_anchor
        self.projector = projector
        
        # Loss functions
        self.infonce_criterion = nn.CrossEntropyLoss()
        
        self.distill_loss_type = distill_loss_type
        if distill_loss_type == 'mse':
            self.distill_criterion = nn.MSELoss()
        elif distill_loss_type == 'cosine':
            self.distill_criterion = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown distill_loss_type: {distill_loss_type}")

        # Task weights (lambda for initialization, buffers for dynamic OL-AUX)
        self.register_buffer('w_mlm', torch.tensor(lambda_mlm))
        self.register_buffer('w_distill', torch.tensor(lambda_distill))
        self.lambda_nce = lambda_nce # Main task is usually fixed at 1.0 or lambda_nce

    def forward(self, input_ids, attention_mask=None, masked_input_ids=None, mlm_labels=None, teacher_embeddings=None, use_ol_aux=False):
        # Determine the batch structure based on your targets
        num_embeddings_per_group = 1 + self.num_targets_per_anchor
        actual_batch_size = input_ids.shape[0] // num_embeddings_per_group
        
        # We need the indices of the anchors to slice the inputs
        # Anchors are at indices 0, 11, 22, etc. (assuming 10 targets)
        anchor_indices = torch.arange(0, input_ids.shape[0], num_embeddings_per_group, device=input_ids.device)

        # ==========================================
        # PASS 1: Masked Language Modeling (MLM on Anchors ONLY)
        # ==========================================
        mlm_loss = torch.tensor(0.0, device=input_ids.device)
        if masked_input_ids is not None and mlm_labels is not None:
            # Slicing: Only pass the anchors through the MASKED pass
            masked_anchor_ids = masked_input_ids[anchor_indices]
            masked_anchor_attention = attention_mask[anchor_indices]
            anchor_mlm_labels = mlm_labels[anchor_indices]

            outputs_mlm = self.student.bert(input_ids=masked_anchor_ids, attention_mask=masked_anchor_attention)
            token_embeddings_mlm = outputs_mlm.last_hidden_state

            prediction_scores = self.student.cls(token_embeddings_mlm)

            mlm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.student.config.vocab_size), 
                anchor_mlm_labels.view(-1),
                ignore_index=-100
            )

        # ==========================================
        # PASS 2: InfoNCE (CLEAN on ALL inputs)
        # ==========================================
        # Second forward pass using CLEAN inputs
        outputs_clean = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings_clean = outputs_clean.last_hidden_state

        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings_clean.size()).to(token_embeddings_clean.dtype)
        sum_embeddings = torch.sum(token_embeddings_clean * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        if self.projector:
            projected_embeddings = self.projector(mean_pooled_embeddings)
            normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        else:
            normalized_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)

        # ------------------------------------------
        # INFONCE (Main Task - Contrastive matching)
        # ------------------------------------------
        # Reshape into groups for contrastive task
        if normalized_embeddings.shape[0] % num_embeddings_per_group == 0:
            grouped_embeddings = normalized_embeddings.view(actual_batch_size, num_embeddings_per_group, -1)
            anchor_embeddings = grouped_embeddings[:, 0, :]
            target_embeddings = grouped_embeddings[:, 1:, :]

            # Calculate Cosine Similarities (Student-internal)
            student_scores = torch.einsum('bh,bkh->bk', anchor_embeddings, target_embeddings)

            # InfoNCE Loss
            ground_truth_labels = torch.zeros(actual_batch_size, dtype=torch.long, device=student_scores.device)
            nce_loss = self.infonce_criterion(student_scores, ground_truth_labels)
        else:
            # Fallback if batch is not a multiple (should not happen with our collator)
            anchor_embeddings = normalized_embeddings[anchor_indices]
            nce_loss = torch.tensor(0.0, device=input_ids.device)
            student_scores = None

        # ------------------------------------------
        # DISTILLATION (Aux Task - Feature matching on Anchors ONLY)
        # ------------------------------------------
        distill_loss = torch.tensor(0.0, device=input_ids.device)
        if teacher_embeddings is not None:
            # Slice teacher embeddings to match our anchors
            if teacher_embeddings.shape[0] == input_ids.shape[0]:
                anchor_teacher_embeddings = teacher_embeddings[anchor_indices]
            else:
                anchor_teacher_embeddings = teacher_embeddings
                
            # Normalize teacher embeddings for cosine comparison
            t_embeddings = F.normalize(anchor_teacher_embeddings, p=2, dim=-1)
            
            # Use the anchor_embeddings we already extracted from the clean pass
            # (Calculated in the InfoNCE block above)
            # If batch was not a multiple, we might need to slice manually
            if student_scores is None:
                anchor_embeddings = normalized_embeddings[anchor_indices]

            if self.distill_loss_type == 'mse':
                distill_loss = self.distill_criterion(anchor_embeddings, t_embeddings)
            elif self.distill_loss_type == 'cosine':
                target = torch.ones(anchor_embeddings.size(0), device=anchor_embeddings.device)
                distill_loss = self.distill_criterion(anchor_embeddings, t_embeddings, target)

        # ==========================================
        # COMBINED LOSS (OL-AUX Dynamic Weighting)
        # ==========================================
        # We extract the python float values from the buffers to prevent PyTorch from
        # adding the buffer tensors to the computation graph. This avoids the 
        # "modified by an inplace operation" RuntimeError when we update them later.
        w_m = self.w_mlm.item() if isinstance(self.w_mlm, torch.Tensor) else self.w_mlm
        w_d = self.w_distill.item() if isinstance(self.w_distill, torch.Tensor) else self.w_distill

        if use_ol_aux:
            # We use standard linear combination here to prevent exploding gradients (NaNs)
            # that occur when using torch.log() on losses that approach zero.
            total_loss = (self.lambda_nce * nce_loss) + \
                         (w_m * mlm_loss) + \
                         (w_d * distill_loss)
        else:
            # Standard linear combination if OL-AUX is disabled
            total_loss = (self.lambda_nce * nce_loss) + \
                         (w_m * mlm_loss) + \
                         (w_d * distill_loss)

        return {
            "loss": total_loss,
            "nce_loss": nce_loss,
            "mlm_loss": mlm_loss,
            "distill_loss": distill_loss,
            "logits": student_scores,
            "w_mlm": self.w_mlm,
            "w_distill": self.w_distill
        }