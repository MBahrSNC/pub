import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# Constants
MODEL_NAME = "stabilityai/stablelm-2-zephyr-1_6b"  # Hugging Face model name
DATASET_NAME = "glaiveai/glaive-function-calling-v2"        # Dataset name
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 2  # Adjust this to fit larger batches into memory
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Do not assume a specific architecture like embeddings or base_model
    # Instead, freeze the first few layers based on available layers

    # Check for layers in the transformer or model
    if hasattr(model, 'transformer'):
        print("Freezing first layers of model.transformer")
        for param in model.transformer.h[:6].parameters():  # Freeze first 6 layers
            param.requires_grad = False
    elif hasattr(model, 'model'):
        print("Freezing first layers of model.model")
        for param in model.model.layers[:6].parameters():  # Freeze first 6 layers
            param.requires_grad = False
    else:
        print("Freezing last 200 parameters as a fallback")
        # Fallback: freeze last few parameters if structure is unknown
        for param in list(model.parameters())[-200:]:
            param.requires_grad = False

    model.resize_token_embeddings(len(tokenizer))
    return model.to(DEVICE), tokenizer

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["chat"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return tokenized_dataset

def create_dataloader(tokenized_dataset):
    return DataLoader(tokenized_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

def train_model(model, dataloader, tokenizer):  # Add tokenizer as argument
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
            model.save_pretrained("mini-llm2")
            tokenizer.save_pretrained("mini-llm2")  # Save the tokenizer as well

def main():
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_dataset(DATASET_NAME)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    train_dataloader = create_dataloader(tokenized_dataset)
    train_model(model, train_dataloader, tokenizer)  # Pass tokenizer to train_model

if __name__ == "__main__":
    main()
