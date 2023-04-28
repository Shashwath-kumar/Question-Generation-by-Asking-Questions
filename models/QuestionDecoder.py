import torch
import torch.nn as nn
from transformers import AutoModel

class QuestionDecoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(QuestionDecoder, self).__init__()
        self.decoder = AutoModel.from_pretrained(pretrained_model_name).decoder
        
    def forward(self, input_ids, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        outputs = self.decoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask)
        contextual_embeddings = outputs.last_hidden_state
        return contextual_embeddings