from transformers import AutoModel, AutoTokenizer
from models.tokenizer import AsmTokenizer
from models.bert import BERT
import torch
import os
import torch.nn.functional as F

def simulate_BERTDataset_without_masking(data_pairs, tokenizer, seq_len):
    bert_inputs = []

    for t1_char_str, t2_char_str in data_pairs:
        t1_tokens = tokenizer.encode(t1_char_str)
        t2_tokens = tokenizer.encode(t2_char_str)
        t1_padded = t1_tokens[:seq_len] + [tokenizer.vocab['[PAD]']] * (seq_len - len(t1_tokens))
        t2_padded = t2_tokens[:seq_len] + [tokenizer.vocab['[PAD]']] * (seq_len - len(t2_tokens))
        bert_input = [tokenizer.vocab['[CLS]']] + t1_padded + [tokenizer.vocab['[SEP]']] + t2_padded + [tokenizer.vocab['[SEP]']]
        bert_inputs.append(bert_input)

    return torch.tensor(bert_inputs, dtype=torch.short)

class EncoderModel:
    def __init__(self, 
                model_type, 
                device, 
                data_dir,
                seq_length=None,
                max_len=None,
        ):
        self.device = device
        self.model_type = model_type
        
        if model_type == 'clap':
            self.tokenizer = AutoTokenizer.from_pretrained('hustcw/clap-asm', trust_remote_code=True)
            self.model = AutoModel.from_pretrained('hustcw/clap-asm', trust_remote_code=True).to(device)

        else:
            self.tokenizer = AsmTokenizer(vocab_file=os.path.join(data_dir, f"baseline-vocab.txt"))
            self.model = BERT(
                vocab_size=len(self.tokenizer.vocab),
                d_model=128,
                n_layers=2,
                heads=1,
                dropout=0.1,
                device=device
            )

            if model_type == 'baseline':
                self.model.load_state_dict(torch.load(os.path.join(data_dir, f'baseline-model'), map_location=torch.device('cpu')))
            elif model_type == 'distil':
                self.model.load_state_dict(torch.load(os.path.join(data_dir, f'distil-embedding-model.pt'), map_location=torch.device('cpu')))
            elif model_type == 'ranking':
                self.model.load_state_dict(torch.load(os.path.join(data_dir, f'distil-ranking-model.pt'), map_location=torch.device('cpu')))

            self.model = self.model.to(device)
           
            self.seq_length = seq_length
            self.total_seq_len = 3 + (seq_length * 2)
            self.max_length = max_len
            self.dynamic_max_len = None
            self.pad_token_id = 0

    def tokenize_data(self, data):
        if self.model_type == 'clap':
            return self.tokenizer(data, padding=True, return_tensors='pt')
        
        else:
            tokenized_data = []
            dynamic_max_len = 0

            for data_pair in data:
                data_pairs_tokenize = simulate_BERTDataset_without_masking(data_pair, self.tokenizer, self.seq_length)
                tokenized_data.append(data_pairs_tokenize)

                # get dynamic len of that batch
                if len(data_pairs_tokenize) > dynamic_max_len:
                    dynamic_max_len = len(data_pairs_tokenize)

            self.dynamic_max_len = min(dynamic_max_len, self.max_length)
            
            final_tokens = []
            final_attention_masks = []

            for tokens in tokenized_data:
                tokens = tokens[:self.dynamic_max_len]  # truncate if needed

                # pad if shorter than dynamic_max_len
                if tokens.shape[0] < dynamic_max_len:
                    pad_len = self.dynamic_max_len - tokens.shape[0]
                    pad_tensor = torch.full((pad_len, tokens.shape[1]), self.pad_token_id, dtype=torch.long)
                    tokens = torch.cat([tokens, pad_tensor], dim=0)

                final_tokens.append(tokens)
                attention_mask = (tokens != self.pad_token_id).any(dim=1).long()
                final_attention_masks.append(attention_mask)

            # stack into batch tensors
            final_tokens = torch.stack(final_tokens)
            final_attention_masks = torch.stack(final_attention_masks)

        return {
            'input_ids': final_tokens,
            'attention_mask': final_attention_masks,
        }



    def compute_embeddings(self, data):
        inputs = self.tokenize_data(data)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.model_type == 'clap':
                asm_embeddings = self.model(**inputs)

                return asm_embeddings.cpu().numpy()
            
            else:
                current_dataloader_batch_size = len(inputs['input_ids'])

                reshaped_input_for_encode = inputs['input_ids'].view(
                    current_dataloader_batch_size * self.dynamic_max_len,
                    self.total_seq_len
                )

                # encode all parts
                encoded_all_parts = self.model.encode(reshaped_input_for_encode)

                # reshape back to [batch_size, max_len, bert_dimension]
                encoded_parts_batched_view = encoded_all_parts.view(
                    current_dataloader_batch_size,
                    self.dynamic_max_len,
                    128
                )

                # masked summation so we discard padded instructions
                attention_mask_expanded = inputs['attention_mask'].unsqueeze(-1)
                masked_encoded_parts = encoded_parts_batched_view * attention_mask_expanded
                asm_summed_function_embeddings = torch.sum(masked_encoded_parts, dim=1)

                # mean pooling
                valid_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-5)
                asm_embeddings = asm_summed_function_embeddings / valid_counts

                # normalize
                asm_embeddings = F.normalize(asm_embeddings, dim=1)

                return asm_embeddings.cpu().numpy()