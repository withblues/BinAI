import torch.nn as nn
import torch.nn.functional as F
import torch


# class StudentWithProjector(nn.Module):
#     def __init__(self, student_model, teacher_dim, loss_fn):
#         super().__init__()
#         self.student = student_model
#         self.projector = nn.Linear(self.student.config.hidden_size, teacher_dim)

#         self.loss_fn = loss_fn
#         if loss_fn == 'mse':
#             self.criterion = nn.MSELoss()
#         elif loss_fn == 'cosine':
#             self.criterion = nn.CosineEmbeddingLoss()

#     def forward(self, input_ids, attention_mask=None, labels=None):
#         # student model forward pass
#         outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
#         s_cls = outputs.last_hidden_state[:, 0, :]

#         # project student to teacher
#         s_proj = self.projector(s_cls)
#         s_proj = F.normalize(s_proj, p=2, dim=-1)

#         loss = None
#         if labels is not None:
#             # mse
#             if self.loss_fn == 'mse':
#                 loss = self.criterion(s_proj, labels)
            
#             # cosine embedding loss
#             elif self.loss_fn == 'cosine':
#                 target = torch.ones(s_proj.size(0), device=s_proj.device)

#                 loss = self.criterion(s_proj, labels, target)

#         return {
#             'loss': loss,
#             'logits': s_proj
#         }
    
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

