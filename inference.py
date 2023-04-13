import heapq
import torch
from typing import List
from final_model import QuestionGenerationModel
from transformers import T5Tokenizer
from dataset_utils import get_test_dataset
from tqdm import tqdm

class BeamSearch:
    def __init__(self, model, tokenizer, device, beam_size=5, max_len=10):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.device = device

    def search(self, batch) -> List[str]:
        self.model.eval()

        with torch.no_grad():
            # Initialize the list of hypotheses
            start_token_id = self.tokenizer.eos_token_id
            hypotheses = [([4073], 0.0)]
            print(hypotheses)

            # Iterate until the maximum sequence length is reached
            for _ in range(self.max_len):
                new_hypotheses = []

                for i, (hypothesis, score) in enumerate(hypotheses):
                    # Add hypothesis tokens to the input
                    # current_input_ids = torch.cat([qg_input_ids, torch.tensor(hypothesis, dtype=torch.long).unsqueeze(0)], dim=-1).to(self.model.device)
                    
                    batch['question_input_ids'] = torch.tensor(hypothesis + [self.tokenizer.pad_token_id], dtype=torch.long).unsqueeze(0).to(device)
                    # Get the logits from the model
                    # logits, _, _, _, _, _, _ = self.model(**batch)
                    logits = self.model(**batch)
                    last_logits = logits[0, -1]
                    print('asasdasdddas', logits.shape, batch['question_input_ids'].shape)

                    # Calculate the probabilities and pick the top k (beam_size) candidates
                    topk_prob, topk_indices = torch.topk(torch.softmax(last_logits, dim=-1), self.beam_size)

                    # Add the new tokens to the hypotheses and update their scores
                    for i in range(self.beam_size):
                        new_hypothesis = hypothesis + [topk_indices[i].item()]
                        new_score = score + torch.log(topk_prob[i]).item()
                        new_hypotheses.append((new_hypothesis, new_score))

                # Keep the top k (beam_size) hypotheses
                hypotheses = heapq.nlargest(self.beam_size, new_hypotheses, key=lambda x: x[1])
                # print(hypotheses)
                # print(self.tokenizer.decode(max(hypotheses, key=lambda x: x[1])[0]))

                # Check if all hypotheses have ended
                if all(self.tokenizer.eos_token_id in hyp[1:] for hyp, _ in hypotheses):
                    break

        # Decode the hypothesis with the highest score
        best_hypothesis, _ = max(hypotheses, key=lambda x: x[1])
        generated_question = self.tokenizer.decode(best_hypothesis, skip_special_tokens=True)

        return generated_question

if __name__ == '__main__':
    pretrained_t5_name = 't5-small'
    d_model = 512 # for t5-small

    bsize = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = QuestionGenerationModel(pretrained_t5_name, d_model, device)
    model.load_state_dict(torch.load('best_model_4_gss_el2.pt', map_location=device))
    model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_name, model_max_length = 512)
    val_dataloader = get_test_dataset(bsize)

    beamsearch = BeamSearch(model, tokenizer, device)
    for batch in tqdm(val_dataloader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        question = beamsearch.search(batch)
        print(question)
