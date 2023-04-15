import heapq
import torch
from typing import List
from final_model import QuestionGenerationModel
from transformers import AutoTokenizer
from dataset_utils import get_test_dataset
from tqdm import tqdm
import config

class BeamSearch:
    def __init__(self, model, tokenizer, device, beam_size=10, max_len=50):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.device = device

    def search(self, batch) -> List[str]:
        self.model.eval()

        with torch.no_grad():
            # Initialize the list of hypotheses
            start_token_id = self.tokenizer.bos_token_id
            hypotheses = [([start_token_id], 0.0)]

            # Iterate until the maximum sequence length is reached
            for _ in range(self.max_len):
                new_hypotheses = []

                for i, (hypothesis, score) in enumerate(hypotheses):
                    # Add hypothesis tokens to the input
                    batch['question_input_ids'] = torch.tensor(hypothesis + [self.tokenizer.eos_token_id]).unsqueeze(0).to(device)
                    # Get the logits from the model
                    logits, _, _, _, _, _, _ = self.model(**batch)
                    last_logits = logits[0, -1]

                    # Calculate the probabilities and pick the top k (beam_size) candidates
                    topk_prob, topk_indices = torch.topk(torch.softmax(last_logits, dim=-1), self.beam_size)

                    # Add the new tokens to the hypotheses and update their scores
                    for i in range(self.beam_size):
                        new_hypothesis = hypothesis + [topk_indices[i].item()]
                        new_score = score + torch.log(topk_prob[i]).item()
                        new_hypotheses.append((new_hypothesis, new_score))

                # Keep the top k (beam_size) hypotheses
                hypotheses = heapq.nlargest(self.beam_size, new_hypotheses, key=lambda x: x[1])

                # Check if all hypotheses have ended
                if all(self.tokenizer.eos_token_id in hyp for hyp, _ in hypotheses):
                    break

        # Decode the hypothesis with the highest score
        best_hypothesis, _ = max(hypotheses, key=lambda x: x[1])
        generated_question = self.tokenizer.decode(best_hypothesis, skip_special_tokens=True)

        return generated_question

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = QuestionGenerationModel(config.model_name, config.d_model, device)
    model.load_state_dict(torch.load('bart_model_4_gs_polayer.pt', map_location=device))
    # model.load_state_dict(torch.load('bart_model_4_gs_nomaxout.pt', map_location=device))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, model_max_length = 512)
    val_dataloader = get_test_dataset(config.batch_size)

    beamsearch = BeamSearch(model, tokenizer, device)
    for batch in tqdm(val_dataloader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        question = beamsearch.search(batch)
        print(question)
