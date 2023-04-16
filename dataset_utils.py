import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
import config

tokenizer = AutoTokenizer.from_pretrained(config.model_name, model_max_length = 512)


def tokenize_and_preprocess(example):
    passage = example['context'].strip()
    question = example['question'].strip()
    answer_text = example['answers']['text'][0].strip()  # Use the first answer in the list

    # Tokenize the passage, question, and answer_text
    passage_tokens = tokenizer.tokenize(passage)
    question_tokens = tokenizer.tokenize(question)
    answer_tokens = tokenizer.tokenize(answer_text)

    question_input_ids = tokenizer(question)['input_ids']

    answer_start = example['answers']['answer_start'][0]
    answer_end = answer_start + len(answer_text) - 1

    # Find the start and end token indices of the answer within the tokenized passage
    token_start = len(tokenizer(passage[:answer_start])['input_ids']) - 1
    token_end = len(tokenizer(passage[:answer_end])['input_ids']) - 1
    # Create input IDs tensor
    qg_input_ids = tokenizer(passage +'</s><s>'+ answer_text)['input_ids']
    # Create task IDs tensor (0 for question generation, 1 for question answering, 2 for KD)
    task_id = 0  # Question generation
    qg_task_ids = torch.tensor([task_id] * len(qg_input_ids))
    # Create segment IDs tensor (0 for passage, 1 for answer, 2 for question)
    qg_segment_ids = torch.tensor([0] * (len(passage_tokens)+2) + [1] * (len(answer_tokens)+2))

    # Create input_ids for AQ
    qa_input_ids = tokenizer(passage + '</s><s>' +question)['input_ids']
    # Create task IDs tensor (0 for question generation, 1 for question answering, 2 for KD)
    task_id = 1  # Question answering
    qa_task_ids = torch.tensor([task_id] * len(qa_input_ids))
    # Create segment IDs tensor (0 for passage, 2 for question)
    qa_segment_ids = torch.tensor([0] * (len(passage_tokens)+2) + [2] * (len(question_tokens)+2))

    # Create input_ids for KD
    kd_input_ids = tokenizer(passage)['input_ids']
    # Create task IDs tensor (0 for question generation, 1 for question answering, 2 for KD)
    task_id = 2  # Knowledge distillation
    kd_task_ids = torch.tensor([task_id] * len(kd_input_ids))
    # Create segment IDs tensor (0 for passage)
    kd_segment_ids = torch.tensor([0] * (len(passage_tokens)+2))

    return {
        'passage_tokens': passage_tokens,
        'question_tokens': question_tokens,
        'answer_tokens': answer_tokens,
        'token_start': token_start,
        'token_end': token_end,
        'qg_input_ids': qg_input_ids,
        'qg_task_ids': qg_task_ids,
        'qg_segment_ids': qg_segment_ids,
        'qa_input_ids': qa_input_ids,
        'qa_task_ids': qa_task_ids,
        'qa_segment_ids': qa_segment_ids,
        'kd_input_ids': kd_input_ids,
        'kd_task_ids': kd_task_ids,
        'kd_segment_ids': kd_segment_ids,
        'question_input_ids': question_input_ids
    }

def pad_tokens(token_lists, padding_token: str = '<pad>'):
    max_length = max(len(tokens) for tokens in token_lists)
    
    padded_tokens = []
    for tokens in token_lists:
        padded = tokens + [padding_token] * (max_length - len(tokens))
        padded_tokens.append(padded)
    
    return padded_tokens

def custom_collate_fn(batch):
    # Separate the batch into individual fields
    passage_tokens = [item['passage_tokens'] for item in batch]
    question_tokens = [item['question_tokens'] for item in batch]
    answer_tokens = [item['answer_tokens'] for item in batch]

    passage_tokens = pad_tokens(passage_tokens)
    question_tokens = pad_tokens(question_tokens)
    answer_tokens = pad_tokens(answer_tokens)
    
    # Separate the input data into separate lists
    qg_input_ids = [torch.tensor(item['qg_input_ids']) for item in batch]
    qg_task_ids = [torch.tensor(item['qg_task_ids']) for item in batch]
    qg_segment_ids = [torch.tensor(item['qg_segment_ids']) for item in batch]
    qa_input_ids = [torch.tensor(item['qa_input_ids']) for item in batch]
    qa_task_ids = [torch.tensor(item['qa_task_ids']) for item in batch]
    qa_segment_ids = [torch.tensor(item['qa_segment_ids']) for item in batch]
    kd_input_ids = [torch.tensor(item['kd_input_ids']) for item in batch]
    kd_task_ids = [torch.tensor(item['kd_task_ids']) for item in batch]
    kd_segment_ids = [torch.tensor(item['kd_segment_ids']) for item in batch]
    question_input_ids = [torch.tensor(item['question_input_ids']) for item in batch]
    

    # Pad the sequences
    qg_input_ids = pad_sequence(qg_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    qg_task_ids = pad_sequence(qg_task_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    qg_segment_ids = pad_sequence(qg_segment_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    qa_input_ids = pad_sequence(qa_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    qa_task_ids = pad_sequence(qa_task_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    qa_segment_ids = pad_sequence(qa_segment_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    kd_input_ids = pad_sequence(kd_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    kd_task_ids = pad_sequence(kd_task_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    kd_segment_ids = pad_sequence(kd_segment_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    question_input_ids = pad_sequence(question_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    token_start = torch.tensor([torch.tensor(item['token_start']) for item in batch])
    token_end = torch.tensor([torch.tensor(item['token_end']) for item in batch])

    return {
        'passage_tokens': passage_tokens,
        'question_tokens': question_tokens,
        'answer_tokens': answer_tokens,
        'token_start': token_start,
        'token_end': token_end,
        'qg_input_ids': qg_input_ids,
        'qg_task_ids': qg_task_ids,
        'qg_segment_ids': qg_segment_ids,
        'qa_input_ids': qa_input_ids,
        'qa_task_ids': qa_task_ids,
        'qa_segment_ids': qa_segment_ids,
        'kd_input_ids': kd_input_ids,
        'kd_task_ids': kd_task_ids,
        'kd_segment_ids': kd_segment_ids,
        'question_input_ids': question_input_ids
    }


def get_train_dataset(bsize = 32):
    train_dataset = load_dataset("squad", split="train")
    train_dataset = train_dataset.map(tokenize_and_preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=bsize, shuffle=False, collate_fn= custom_collate_fn)
    return train_dataloader

def get_test_dataset(bsize = 32):
    test_dataset = load_dataset("squad", split="validation")
    test_dataset = test_dataset.map(tokenize_and_preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=bsize, shuffle=False, collate_fn= custom_collate_fn)
    return test_dataloader

def get_distributed_dataset(bsize, world_size, rank):
    train_dataset = load_dataset("squad", split="train")
    train_dataset = train_dataset.map(tokenize_and_preprocess)
    
    test_dataset = load_dataset("squad", split="validation")
    test_dataset = test_dataset.map(tokenize_and_preprocess)

    train_sampler = DistributedSampler(train_dataset, world_size, rank)
    val_sampler = DistributedSampler(test_dataset, world_size, rank)
    train_dataloader = DataLoader(train_dataset, batch_size=bsize, sampler=train_sampler, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(test_dataset, batch_size=bsize, sampler=val_sampler, collate_fn=custom_collate_fn)

    return train_dataloader, val_dataloader

if __name__ == '__main__':
    test_dataset = load_dataset("squad", split="validation")
    test_dataset = test_dataset.map(tokenize_and_preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn= custom_collate_fn)
    # test_dataloader = get_test_dataset(2)

    for i, sample in enumerate(test_dataloader):
        print((sample['question_input_ids']))
        if i ==50:
            break