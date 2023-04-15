import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class QuestionAnsweringOutputLayer(nn.Module):
    def __init__(self, d_model):
        super(QuestionAnsweringOutputLayer, self).__init__()
        self.start_logits = nn.Linear(d_model, 1)
        self.end_logits = nn.Linear(d_model * 2, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, contextual_embeddings, attention_mask=None):
        # Predict start indices
        start_scores = self.start_logits(contextual_embeddings).squeeze(-1)
        if attention_mask is not None:
            start_scores = start_scores.masked_fill(~attention_mask, float('-inf'))
        start_probs = self.softmax(start_scores)

        # Predict end indices based on the start index
        best_start_idx = torch.argmax(start_probs, dim=-1, keepdim=True)
        
        # mask out the positions before the start index to avoid generating invalid answer spans
        start_mask = torch.ones_like(attention_mask)
        start_mask.scatter_(1, best_start_idx, 0)
        cumulative_mask = torch.cumsum(start_mask, dim=1)
        invalid_positions_mask = cumulative_mask == 0

        best_start_embeddings = contextual_embeddings.gather(1, best_start_idx.unsqueeze(-1).repeat(1, 1, contextual_embeddings.size(-1)))
        concatenated_embeddings = torch.cat((contextual_embeddings, best_start_embeddings.repeat(1, contextual_embeddings.size(1), 1)), dim=-1)
        end_scores = self.end_logits(concatenated_embeddings).squeeze(-1)
        
        if attention_mask is not None:
            end_scores = end_scores.masked_fill((~attention_mask) | invalid_positions_mask, float('-inf'))
        end_probs = self.softmax(end_scores)

        return start_scores, end_scores, start_probs, end_probs



class QuestionGenerationOutputLayer(nn.Module):
    def __init__(self, model_name):
        super(QuestionGenerationOutputLayer, self).__init__()
        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.output_layer = self.pretrained_model.lm_head

    def forward(self, decoder_output):
        logits = self.output_layer(decoder_output)        
        return logits

