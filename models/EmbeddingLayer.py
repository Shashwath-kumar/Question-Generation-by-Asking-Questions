
import torch.nn as nn
import math
from transformers import AutoModel

class EmbeddingLayer(nn.Module):
    def __init__(self, pretrained_bart_name, d_model=512):
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.word_embedding = AutoModel.from_pretrained(pretrained_bart_name).shared

        self.task_embedding = nn.Embedding(3, d_model)  # Create a new embedding layer for task embeddings
        self.segment_embedding = nn.Embedding(3, d_model)  # Create a new embedding layer for segment embeddings


    def forward(self, input_ids, task_ids, segment_ids):
        word_embeds = self.word_embedding(input_ids)
        task_embeds = self.task_embedding(task_ids)
        segment_embeds = self.segment_embedding(segment_ids)

        embeds = word_embeds + (task_embeds + segment_embeds) / math.sqrt(self.d_model)
        return embeds
