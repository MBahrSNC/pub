import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# Constants
MODEL_NAME = "stabilityai/stablelm-2-zephyr-1_6b"  # Replace with the actual Hugging Face model name
DATASET_NAME = "fka/awesome-chatgpt-prompts"    # Replace with the actual dataset name from Hugging Face
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 2  # Adjust this to fit larger batches into memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Find and unfreeze specific layers based on model architecture
    # Try to access the correct attribute based on model structure
    if hasattr(model, 'transformer'):  # Common for GPT-style models
        for param in model.transformer.h[-2:].parameters():  # Adjust number of layers to unfreeze
            param.requires_grad = True
    elif hasattr(model, 'model'):  # Some models use 'model' to encapsulate transformer layers
        for param in model.model.layers[-2:].parameters():  # Adjust number of layers to unfreeze
            param.requires_grad = True
    else:
        # Fallback: unfreeze the last few layers directly from the model parameters
        for param in list(model.parameters())[-200:]:  # Adjust the number of parameters you want to unfreeze
            param.requires_grad = True

    model.resize_token_embeddings(len(tokenizer))
    return model.to(DEVICE), tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return tokenized_dataset

def create_dataloader(tokenized_dataset):
    return DataLoader(tokenized_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

def train_model(model, dataloader):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scaler = GradScaler()
    model.train()
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for step, batch in enumerate(dataloader):
            inputs = {key: val.to(DEVICE) for key, val in batch.items() if key in ['input_ids', 'attention_mask']}
            labels = inputs['input_ids']

            optimizer.zero_grad()
            with autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
            
            running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Save best model based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained("best_fine_tuned_stablelm_model")
            tokenizer.save_pretrained("best_fine_tuned_stablelm_model")

def main():
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset(DATASET_NAME)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    train_dataloader = create_dataloader(tokenized_dataset)
    train_model(model, train_dataloader)

if __name__ == "__main__":
    main()
