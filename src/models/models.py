import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.empty_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]
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
    def __init__(self, student_model, num_targets_per_anchor, projector=None, lambda_mlm=1.0, lambda_nce=1.0, lambda_distill=1.0, distill_loss_type='mse', temperature_init=0.07):
        super().__init__()
        # student_model should be initialized as BertForMaskedLM
        self.student = student_model 
        self.num_targets_per_anchor = num_targets_per_anchor
        self.projector = projector
        
        # Fixed temperature for InfoNCE (not learnable — prevents optimizer from escaping)
        self.temperature = temperature_init
        
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

            # Calculate Cosine Similarities scaled by fixed temperature
            student_scores = torch.einsum('bh,bkh->bk', anchor_embeddings, target_embeddings) / self.temperature

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
        if use_ol_aux:
            # Hide buffers from torch.compile. Trainer overrides this loss anyway.
            total_loss = nce_loss + mlm_loss + distill_loss
        else:
            w_m = self.w_mlm.detach().clone() if isinstance(self.w_mlm, torch.Tensor) else self.w_mlm
            w_d = self.w_distill.detach().clone() if isinstance(self.w_distill, torch.Tensor) else self.w_distill
            
            # Standard linear combination if OL-AUX is disabled
            total_loss = (self.lambda_nce * nce_loss) + \
                         (w_m * mlm_loss) + \
                         (w_d * distill_loss)

        return {
            "loss": total_loss,
            "logits": student_scores,
            "nce_loss": nce_loss,
            "nce_loss_scaled": self.lambda_nce * nce_loss,
            "distill_loss": distill_loss,
            "distill_loss_scaled": self.lambda_distill * distill_loss,
            "mlm_loss": mlm_loss,
            "mlm_loss_scaled": self.lambda_mlm * mlm_loss
        }

class StudentWithInBatchCosine(nn.Module):
    def __init__(self, student_model, projector=None, distill_loss_type='mse', temperature=0.05, distill_temperature=2.0, distill_topk=32):
        super().__init__()
        self.student = student_model
        self.projector = projector
        self.distill_loss_type = distill_loss_type
        self.temperature = temperature
        self.distill_temperature = distill_temperature
        self.distill_topk = distill_topk
        self.criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask=None, teacher_embeddings=None, binary_names=None, function_names=None, **kwargs):
        # 1. Forward pass
        outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # 2. Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        if self.projector:
            projected_embeddings = self.projector(mean_pooled_embeddings)
            student_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        else:
            student_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)

        loss = None
        predicted_scores = None

        if teacher_embeddings is not None:
            teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)
            
            # Calculate BxB similarity matrices
            student_sims = torch.matmul(student_embeddings, student_embeddings.T)
            teacher_sims = torch.matmul(teacher_embeddings, teacher_embeddings.T)

            if self.distill_loss_type == 'kl':
                teacher_logits = teacher_sims / self.distill_temperature
                student_logits = student_sims / self.distill_temperature
                
                # Mask out diagonal (self-similarity)
                diag_mask = torch.eye(student_logits.shape[0], dtype=torch.bool, device=student_logits.device)
                teacher_logits.masked_fill_(diag_mask, -1e9)
                student_logits.masked_fill_(diag_mask, -1e9)
                
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                
                loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            elif self.distill_loss_type == 'topk_kl':
                teacher_logits = teacher_sims / self.distill_temperature
                student_logits = student_sims / self.distill_temperature
                
                diag_mask = torch.eye(student_logits.shape[0], dtype=torch.bool, device=student_logits.device)
                teacher_logits.masked_fill_(diag_mask, -1e9)
                student_logits.masked_fill_(diag_mask, -1e9)
                
                k = min(self.distill_topk, teacher_logits.size(-1))
                topk_teacher_logits, topk_indices = torch.topk(teacher_logits, k, dim=-1)
                topk_student_logits = torch.gather(student_logits, dim=-1, index=topk_indices)
                
                teacher_probs = F.softmax(topk_teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(topk_student_logits, dim=-1)
                
                loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            elif self.distill_loss_type == 'pairwiserank':
                teacher_logits = teacher_sims / self.distill_temperature
                student_logits = student_sims / self.distill_temperature
                
                diag_mask = torch.eye(student_logits.shape[0], dtype=torch.bool, device=student_logits.device)
                teacher_logits.masked_fill_(diag_mask, -1e9)
                student_logits.masked_fill_(diag_mask, -1e9)
                
                k = min(self.distill_topk, teacher_logits.size(-1))
                topk_teacher_logits, topk_indices = torch.topk(teacher_logits, k, dim=-1)
                topk_student_logits = torch.gather(student_logits, dim=-1, index=topk_indices)
                
                idx = torch.triu_indices(k, k, offset=1, device=teacher_logits.device)
                j, m = idx[0], idx[1]
                
                teacher_gap = topk_teacher_logits[:, j] - topk_teacher_logits[:, m]
                student_diff = topk_student_logits[:, j] - topk_student_logits[:, m]
                
                pair_loss = -F.logsigmoid(student_diff)
                loss = (teacher_gap * pair_loss).mean()
            else:
                loss = self.criterion(student_sims, teacher_sims)
                
            predicted_scores = student_sims

        return {
            "loss": loss,
            "logits": predicted_scores
        }

class StudentWithInBatchInfoNCE(nn.Module):
    def __init__(self, student_model, projector=None, temperature=0.05):
        super().__init__()
        self.student = student_model
        self.projector = projector
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, binary_names=None, function_names=None, **kwargs):
        # 1. Forward pass
        outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # 2. Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        if self.projector:
            projected_embeddings = self.projector(mean_pooled_embeddings)
            student_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        else:
            student_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)

        loss = None
        predicted_scores = None

        if binary_names is not None and function_names is not None:
            # We assume batch is structured: first B are anchors, next B are positives
            total_b = student_embeddings.shape[0]
            B = total_b // 2

            # Compute pairwise similarities (2B x 2B)
            sim_matrix = torch.matmul(student_embeddings, student_embeddings.T) / self.temperature
            predicted_scores = sim_matrix

            # Build masking matrix for collisions
            bin_names = np.array(binary_names)
            func_names = np.array(function_names)
            
            # is_name_collision[i, j] = True if they have same bin and func
            same_bin = (bin_names[:, None] == bin_names[None, :])
            same_func = (func_names[:, None] == func_names[None, :])
            is_name_collision = torch.tensor(same_bin & same_func, device=sim_matrix.device)
            
            # is_hash_collision[i, j] = True if they have perfectly identical token sequences
            is_hash_collision = torch.all(input_ids[:, None, :] == input_ids[None, :, :], dim=-1)

            # Combined collision mask
            is_collision = is_name_collision | is_hash_collision

            # Targets: for i < B target is i + B, for i >= B target is i - B
            targets = torch.empty(total_b, dtype=torch.long, device=sim_matrix.device)
            targets[:B] = torch.arange(B, total_b, device=sim_matrix.device)
            targets[B:] = torch.arange(0, B, device=sim_matrix.device)
            
            # We mask out the diagonal (cannot be target for itself)
            sim_matrix.fill_diagonal_(-1e9)
            
            # Mask out collisions (false negatives)
            is_explicit_target = torch.zeros_like(is_collision, dtype=torch.bool)
            is_explicit_target[torch.arange(total_b), targets] = True
            
            mask_out = is_collision & ~is_explicit_target
            sim_matrix.masked_fill_(mask_out, -1e9)

            loss = self.criterion(sim_matrix, targets)

        return {
            "loss": loss,
            "logits": predicted_scores
        }

class StudentWithJointInBatch(nn.Module):
    def __init__(self, student_model, projector=None, temperature=0.05, lambda_nce=1.0, lambda_distill=1.0, lambda_mlm=1.0, distill_loss_type='mse', distill_temperature=2.0, distill_topk=32):
        super().__init__()
        self.student = student_model
        self.projector = projector
        self.temperature = temperature
        self.distill_temperature = distill_temperature
        self.distill_topk = distill_topk
        
        self.lambda_nce = lambda_nce
        self.lambda_distill = lambda_distill
        self.lambda_mlm = lambda_mlm
        self.distill_loss_type = distill_loss_type
        
        self.register_buffer('w_mlm', torch.tensor(lambda_mlm, dtype=torch.float32))
        self.register_buffer('w_distill', torch.tensor(lambda_distill, dtype=torch.float32))
        
        self.infonce_criterion = nn.CrossEntropyLoss()
        
        if self.distill_loss_type == 'embedding_cosine':
            self.distill_criterion = nn.CosineEmbeddingLoss()
        else:
            self.distill_criterion = nn.MSELoss()

    def forward(self, input_ids, attention_mask=None, binary_names=None, function_names=None, teacher_embeddings=None, masked_input_ids=None, mlm_labels=None, **kwargs):
        
        mlm_loss = torch.tensor(0.0, device=input_ids.device)
        
        # --- PASS 1: MLM on Anchor Sequences ---
        if masked_input_ids is not None and mlm_labels is not None and self.lambda_mlm > 0:
            B = input_ids.shape[0] // 2
            
            anchor_masked_input_ids = masked_input_ids[:B]
            anchor_attention_mask = attention_mask[:B]
            anchor_mlm_labels = mlm_labels[:B]
            
            mlm_outputs = self.student(
                input_ids=anchor_masked_input_ids,
                attention_mask=anchor_attention_mask,
                labels=anchor_mlm_labels
            )
            mlm_loss = mlm_outputs.loss

        # --- PASS 2: Clean Forward Pass for InfoNCE & Distillation ---
        outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        if self.projector:
            projected_embeddings = self.projector(mean_pooled_embeddings)
            student_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        else:
            student_embeddings = F.normalize(mean_pooled_embeddings, p=2, dim=-1)

        total_loss = None
        predicted_scores = None
        nce_loss = torch.tensor(0.0, device=input_ids.device)
        distill_loss = torch.tensor(0.0, device=input_ids.device)

        if binary_names is not None and function_names is not None and teacher_embeddings is not None:
            total_b = student_embeddings.shape[0]
            B = total_b // 2

            teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)
            
            student_sims_unscaled = torch.matmul(student_embeddings, student_embeddings.T)
            teacher_sims = torch.matmul(teacher_embeddings, teacher_embeddings.T)
            
            sim_matrix = student_sims_unscaled / self.temperature
            
            total_b = sim_matrix.size(0)
            
            bin_names = np.array(binary_names)
            func_names = np.array(function_names)
            
            same_bin = (bin_names[:, None] == bin_names[None, :])
            same_func = (func_names[:, None] == func_names[None, :])
            is_name_collision = torch.tensor(same_bin & same_func, device=sim_matrix.device)
            is_hash_collision = torch.all(input_ids[:, None, :] == input_ids[None, :, :], dim=-1)
            
            is_collision = is_name_collision | is_hash_collision
            
            targets = torch.zeros(total_b, dtype=torch.long, device=sim_matrix.device)
            targets[:B] = torch.arange(B, total_b, device=sim_matrix.device)
            targets[B:] = torch.arange(0, B, device=sim_matrix.device)
            
            sim_matrix.fill_diagonal_(-1e9)
            
            is_explicit_target = torch.zeros_like(is_collision, dtype=torch.bool)
            is_explicit_target[torch.arange(total_b), targets] = True
            
            mask_out = is_collision & ~is_explicit_target
            sim_matrix.masked_fill_(mask_out, -1e9)

            nce_loss = self.infonce_criterion(sim_matrix, targets)
            predicted_scores = sim_matrix
            
            if self.distill_loss_type == 'kl_retrieval':
                student_logits = student_sims_unscaled / self.distill_temperature
                teacher_logits = teacher_sims / self.distill_temperature
                
                diag_mask = torch.eye(total_b, dtype=torch.bool, device=teacher_logits.device)
                teacher_logits.masked_fill_(diag_mask, -1e9)
                student_logits.masked_fill_(diag_mask, -1e9)
                
                teacher_logits.masked_fill_(mask_out, -1e9)
                student_logits.masked_fill_(mask_out, -1e9)
                
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                
                distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            elif self.distill_loss_type == 'kl':
                teacher_logits = teacher_sims / self.distill_temperature
                student_logits = student_sims_unscaled / self.distill_temperature
                
                diag_mask = torch.eye(total_b, dtype=torch.bool, device=teacher_logits.device)
                teacher_logits.masked_fill_(diag_mask, -1e9)
                student_logits.masked_fill_(diag_mask, -1e9)
                
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                
                distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            elif self.distill_loss_type in ['topk_kl', 'topk_kl_retrieval']:
                teacher_logits = teacher_sims / self.distill_temperature
                student_logits = student_sims_unscaled / self.distill_temperature
                
                diag_mask = torch.eye(total_b, dtype=torch.bool, device=teacher_logits.device)
                teacher_logits.masked_fill_(diag_mask, -1e9)
                student_logits.masked_fill_(diag_mask, -1e9)
                
                if self.distill_loss_type == 'topk_kl_retrieval':
                    teacher_logits.masked_fill_(mask_out, -1e9)
                    student_logits.masked_fill_(mask_out, -1e9)
                    
                k = min(self.distill_topk, teacher_logits.size(-1))
                topk_teacher_logits, topk_indices = torch.topk(teacher_logits, k, dim=-1)
                topk_student_logits = torch.gather(student_logits, dim=-1, index=topk_indices)
                
                teacher_probs = F.softmax(topk_teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(topk_student_logits, dim=-1)
                
                distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            elif self.distill_loss_type in ['pairwiserank', 'pairwiserank_retrieval']:
                teacher_logits = teacher_sims / self.distill_temperature
                student_logits = student_sims_unscaled / self.distill_temperature
                
                diag_mask = torch.eye(total_b, dtype=torch.bool, device=teacher_logits.device)
                teacher_logits.masked_fill_(diag_mask, -1e9)
                student_logits.masked_fill_(diag_mask, -1e9)
                
                if self.distill_loss_type == 'pairwiserank_retrieval':
                    teacher_logits.masked_fill_(mask_out, -1e9)
                    student_logits.masked_fill_(mask_out, -1e9)
                    
                k = min(self.distill_topk, teacher_logits.size(-1))
                topk_teacher_logits, topk_indices = torch.topk(teacher_logits, k, dim=-1)
                topk_student_logits = torch.gather(student_logits, dim=-1, index=topk_indices)
                
                idx = torch.triu_indices(k, k, offset=1, device=teacher_logits.device)
                j, m = idx[0], idx[1]
                
                teacher_gap = topk_teacher_logits[:, j] - topk_teacher_logits[:, m]
                student_diff = topk_student_logits[:, j] - topk_student_logits[:, m]
                
                pair_loss = -F.logsigmoid(student_diff)
                distill_loss = (teacher_gap * pair_loss).mean()
            elif self.distill_loss_type == 'embedding_cosine':
                target = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
                distill_loss = self.distill_criterion(student_embeddings, teacher_embeddings, target)
            elif self.distill_loss_type == 'embedding_mse':
                # Direct MSE on the normalized embeddings
                distill_loss = self.distill_criterion(student_embeddings, teacher_embeddings)
            else:
                distill_loss = self.distill_criterion(student_sims_unscaled, teacher_sims)
            
            total_loss = (self.lambda_nce * nce_loss) + (self.lambda_distill * distill_loss) + (self.lambda_mlm * mlm_loss)

        return {
            "loss": total_loss,
            "logits": predicted_scores,
            "nce_loss": nce_loss,
            "nce_loss_scaled": self.lambda_nce * nce_loss,
            "distill_loss": distill_loss,
            "distill_loss_scaled": self.lambda_distill * distill_loss,
            "mlm_loss": mlm_loss,
            "mlm_loss_scaled": self.lambda_mlm * mlm_loss
        }