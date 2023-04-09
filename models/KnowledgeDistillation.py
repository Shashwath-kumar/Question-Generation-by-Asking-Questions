import torch
import torch.nn as nn
from transformers import T5Model, T5Tokenizer

class KnowledgeDistillation(nn.Module):
    def __init__(self, pretrained_model_name, d_model, vocab_size):
        super(KnowledgeDistillation, self).__init__()
        self.pretrained_model = T5Model.from_pretrained(pretrained_model_name)
        self.Wm = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, input_mask, primal_dual_embeddings):
        # Get the contextual embeddings from the pretrained model
        with torch.no_grad():
            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids = input_ids)
            pretrained_contextual_embeddings = outputs.last_hidden_state

        # Get the output embeddings of the masked words from the primal-dual encoder
        masked_word_embeddings_en = primal_dual_embeddings.masked_select(input_mask.unsqueeze(-1)).view(-1, primal_dual_embeddings.size(-1))

        # Calculate the probability distributions for the masked words using the output matrix Wm
        y_en = self.softmax(self.Wm(masked_word_embeddings_en))

        # Get the output embeddings of the masked words from the pretrained model
        masked_word_embeddings_pre = pretrained_contextual_embeddings.masked_select(input_mask.unsqueeze(-1)).view(-1, pretrained_contextual_embeddings.size(-1))

        # Calculate the probability distributions for the masked words using the output matrix Wm
        y_pre = self.softmax(self.Wm(masked_word_embeddings_pre))

        return y_en, y_pre
