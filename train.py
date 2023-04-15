import torch
import torch.nn as nn
from dataset_utils import get_train_dataset, get_test_dataset
from final_model import QuestionGenerationModel
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import config

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, learning_rate):
    # Move the model to the device
    model = model.to(device)
    # Set up the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    ce_loss = nn.CrossEntropyLoss()

    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = 5000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            logits, start_scores, end_scores, _, _, y_en, y_pre = model(**batch)

            target_ids = batch['question_input_ids'][:, 1:]
            qg_loss = ce_loss(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            answer_start = batch["token_start"]
            answer_end = batch["token_end"]

            qa_loss = ce_loss(start_scores, answer_start) + ce_loss(end_scores, answer_end)

            kd_loss = -torch.sum(y_en * torch.log(y_pre))

            total_loss = qg_loss + 0.8 * qa_loss + 0.15 * kd_loss
            total_loss.backward()

            optimizer.step()
            scheduler.step()
            train_loss += total_loss.item()

        train_loss /= len(train_dataloader)
        print(f"Train Loss: {train_loss}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                logits, start_scores, end_scores, _, _, y_en, y_pre = model(**batch)

                target_ids = batch['question_input_ids'][:, 1:]
                qg_loss = ce_loss(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

                answer_start = batch["token_start"]
                answer_end = batch["token_end"]
                qa_loss = ce_loss(start_scores, answer_start) + ce_loss(end_scores, answer_end)

                kd_loss = -torch.sum(y_en * torch.log(y_pre))

                total_loss = qg_loss + 0.8 * qa_loss + 0.15 * kd_loss
                val_loss += total_loss.item()

        val_loss /= len(val_dataloader)
        print(f"Validation Loss: {val_loss}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.model_save_name}")
            print(f"Best model saved. {config.model_save_name}")

    return best_model

if __name__ == '__main__':

    train_dataloader = get_train_dataset(config.batch_size)
    val_dataloader = get_test_dataset(config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = QuestionGenerationModel(config.model_name, config.d_model, device)
    
    train_model(model, train_dataloader, val_dataloader, device, config.num_epochs, config.lr)