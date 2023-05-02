from transformers import AutoModel

model_name = 'facebook/bart-base'

temp_model = AutoModel.from_pretrained(model_name)
d_model = temp_model.config.d_model
del temp_model

batch_size = 2
num_epochs = 15
lr = 3e-5

model_save_name = 'QG_SQuAD.pt'