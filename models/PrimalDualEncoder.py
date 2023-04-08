import torch.nn as nn
from transformers import T5Model

class PrimalDualEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super(PrimalDualEncoder, self).__init__()
        self.encoder = T5Model.from_pretrained(pretrained_model_name).encoder

    def forward(self, embeddings, attention_mask=None):
        outputs = self.encoder(inputs_embeds=embeddings, attention_mask=attention_mask)
        contextual_embeddings = outputs.last_hidden_state
        return contextual_embeddings
