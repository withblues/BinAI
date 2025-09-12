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
        s_cls = outputs.last_hidden_state[:, 0, :]

        # project student to teacher
        s_proj = self.projector(s_cls)
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
    def __init__(self, student_model):
        super().__init__()
        self.student = student_model
        self.criterion = nn.MSELoss()


    def forward(self, input_ids, attention_mask=None, labels=None):
        # student model forward pass

        print(f'shape of input_ids {input_ids.shape}')
        print(f'shape of attention_mask {attention_mask.shape}')
        outputs = self.student.bert(input_ids=input_ids, attention_mask=attention_mask)
        s_cls = outputs.last_hidden_state[:, 0, :]
        s_cls = F.normalize(s_cls, p=2, dim=-1)
        print(f'output bert model shape {s_cls.shape}')

        # split to anchor and multiple targets
        batch_size = labels.shape[0]
        num_targets = labels.shape[1]

        grouped_embeddings = s_cls.view(batch_size, num_targets + 1, -1)

        anchor_embeddings = grouped_embeddings[:, 0, :]
        target_embeddings = grouped_embeddings[:, 1:, :]
        print(f'embedding anchor {anchor_embeddings.shape}')
        print(f'embedding target {target_embeddings.shape}')

        # caculate cosine similarity 
        predicted_scores = torch.einsum('bh,bkh->bk', anchor_embeddings, target_embeddings)

        print(f'scores {predicted_scores.shape}')
        loss = None
        if labels is not None:
            loss = self.criterion(predicted_scores, labels)

        return {
            "loss": loss,
            "logits": predicted_scores
        }
