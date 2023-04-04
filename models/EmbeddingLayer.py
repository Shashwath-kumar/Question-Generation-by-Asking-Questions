import torch
import torch.nn as nn
from transformers import T5Model

class EmbeddingLayer(nn.Module):
    def __init__(self, pretrained_t5_name, d_model=512):
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.t5 = T5Model.from_pretrained(pretrained_t5_name)
        self.word_embedding = self.t5.shared  # Use T5's word embeddings
        self.task_embedding = nn.Embedding(3, d_model)  # Create a new embedding layer for task embeddings
        self.position_embedding = self.t5.encoder.embed_positions
        self.segment_embedding = nn.Embedding(3, d_model)  # Create a new embedding layer for segment embeddings

    def forward(self, input_ids, task_ids, segment_ids, position_ids=None):
        word_embeds = self.word_embedding(input_ids)
        task_embeds = self.task_embedding(task_ids)
        position_embeds = self.position_embedding(input_ids)
        segment_embeds = self.segment_embedding(segment_ids)

        embeds = word_embeds + task_embeds + position_embeds + segment_embeds
        return embeds