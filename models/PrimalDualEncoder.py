import torch.nn as nn
from transformers import T5Model

class PrimalDualEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(PrimalDualEncoder, self).__init__()
        self.encoder = T5Model.from_pretrained(pretrained_model_name).encoder

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        contextual_embeddings = outputs.last_hidden_state
        return contextual_embeddings
