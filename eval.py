import sacrebleu
import torch
from inference import BeamSearch
from torch.utils.data import DataLoader
from dataset_utils import get_test_dataset
from final_model import QuestionGenerationModel
import config
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

def evaluate(model, tokenizer, device, val_dataloader: DataLoader, beam_size=10, max_len=50) -> float:
    beam_search = BeamSearch(model, tokenizer, device, beam_size=beam_size, max_len=max_len)

    references = []
    hypotheses = []

    for batch in tqdm(val_dataloader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                
        # Generate the question using Beam Search
        generated_question = beam_search.search(batch)

        # Append the generated question to the list of hypotheses
        hypotheses.append(generated_question)

        # Append the true question to the list of references
        true_question = tokenizer.decode(batch['question_input_ids'][0], skip_special_tokens=True)
        references.append([true_question])
        break

    df = pd.DataFrame({'hypothesis':hypotheses, 'references':references})
    df.to_csv('final_questions.csv', index=False)
    # Calculate the BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    bleu_score = bleu.score
    return bleu_score


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = QuestionGenerationModel(config.model_name, config.d_model, device)
    model.load_state_dict(torch.load(f'bart_model_4_new_val.pt'))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, model_max_length = 512)

    val_dataloader = get_test_dataset(bsize=1)

    # Evaluate the model on the validation dataset
    bleu_score = evaluate(model, tokenizer, device, val_dataloader)
    print(f"SacreBLEU Score: {bleu_score}")
