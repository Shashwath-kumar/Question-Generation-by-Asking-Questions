import torch
import torch.nn as nn
from models import EmbeddingLayer, PrimalDualEncoder, QuestionDecoder, QuestionAnsweringOutputLayer, QuestionGenerationOutputLayer, KnowledgeDistillation
from transformers import AutoTokenizer
import random

class QuestionGenerationModel(nn.Module):
    def __init__(self, pretrained_model_name, d_model, device) -> None:
        super(QuestionGenerationModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, model_max_length = 512)
        self.embedding_layer = EmbeddingLayer(pretrained_model_name, d_model)
        self.primal_dual_encoder = PrimalDualEncoder(pretrained_model_name)
        self.qg_question_decoder = QuestionDecoder(pretrained_model_name)
        self.qg_output_layer = QuestionGenerationOutputLayer(pretrained_model_name)
        self.qa_output_layer = QuestionAnsweringOutputLayer(d_model)
        self.kd_layer = KnowledgeDistillation(pretrained_model_name, d_model, self.tokenizer.vocab_size)
        self.device = device

        self.to(device)

    def create_attention_mask(self, input_ids):
        return (input_ids != self.tokenizer.pad_token_id)
    
    def create_distillation_mask(self, tokens, mask_rate=0.15):
        mask = [random.random() < mask_rate for _ in tokens]
        return [False] + mask + [False]


    def forward(self, 
                passage_tokens,
                question_tokens,
                answer_tokens,
                token_start,
                token_end,
                qg_input_ids,
                qg_task_ids,
                qg_segment_ids,
                qa_input_ids,
                qa_task_ids,
                qa_segment_ids,
                kd_input_ids,
                kd_task_ids,
                kd_segment_ids,
                question_input_ids
                ):
        
        # QG Module
        qg_embeddings = self.embedding_layer(qg_input_ids, qg_task_ids, qg_segment_ids)
        qg_attention_mask = self.create_attention_mask(qg_input_ids).to(self.device)
        qg_encoder_outputs = self.primal_dual_encoder(qg_embeddings, qg_attention_mask)
        qg_target_ids = question_input_ids[:, :-1]
        qg_target_attention_mask = self.create_attention_mask(qg_target_ids).to(self.device)
        decoder_outputs = self.qg_question_decoder(input_ids=qg_target_ids,
                                   attention_mask=qg_target_attention_mask,
                                   encoder_hidden_states=qg_encoder_outputs)
        logits = self.qg_output_layer(decoder_outputs)

        # AQ Module
        qa_embeddings = self.embedding_layer(qa_input_ids, qa_task_ids, qa_segment_ids)
        qa_attention_mask = self.create_attention_mask(qa_input_ids).to(self.device)
        qa_encoder_outputs = self.primal_dual_encoder(qa_embeddings, qa_attention_mask)
        start_scores, end_scores, start_probs, end_probs = self.qa_output_layer(qa_encoder_outputs, qa_attention_mask)

        # UG Module
        kd_embeddings = self.embedding_layer(kd_input_ids, kd_task_ids, kd_segment_ids)
        kd_attention_mask = self.create_attention_mask(kd_input_ids).to(self.device)
        kd_encoder_outputs = self.primal_dual_encoder(kd_embeddings, kd_attention_mask)
        distillation_masks = [self.create_distillation_mask(tokens) for tokens in passage_tokens]
        distillation_mask_tensor = torch.stack([torch.tensor(mask, dtype=torch.bool) for mask in distillation_masks]).to(self.device)

        y_en, y_pre = self.kd_layer(kd_input_ids, distillation_mask_tensor, kd_encoder_outputs)

        return logits, start_scores, end_scores, start_probs, end_probs, y_en, y_pre