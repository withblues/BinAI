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