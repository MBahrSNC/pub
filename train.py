import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint_sequential

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize tokenizer and set padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load the PersonaHub dataset
persona_dataset = load_dataset("proj-persona/PersonaHub")

class PersonaDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_point = self.dataset[idx]
        input_text = data_point["input persona"] + " " + data_point["synthesized text"]
        inputs = self.tokenizer.encode(input_text, truncation=True, max_length=self.max_length, padding='max_length')
        return torch.tensor(inputs)

# Parameters
max_seq_length = 50000  # Set sequence length to 50k tokens
batch_size = 1  # Batch size of 1

# Create PersonaHub dataset
persona_data = persona_dataset['train']  # or use 'test', 'validation' based on your requirements
persona_dataset = PersonaDataset(persona_data, tokenizer, max_seq_length)
dataloader = DataLoader(persona_dataset, batch_size=batch_size, shuffle=True)

# Define model, optimizer, etc. as previously discussed
class MokiTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward):
        super(MokiTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src = self.embedding(src) * (self.d_model ** 0.5)
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)
        # Apply gradient checkpointing to save memory
        segments = 4  # Number of segments for checkpointing
        src = checkpoint_sequential(self.transformer.encoder.layers, segments, src)
        tgt = checkpoint_sequential(self.transformer.decoder.layers, segments, tgt)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

# Create directory for moki
os.makedirs("moki", exist_ok=True)

# Example hyperparameters for a 1B parameter model
vocab_size = len(tokenizer)
d_model = 1024  # Reduced model dimension for larger context handling
nhead = 16  # Reduced number of heads
num_encoder_layers = 12  # Reduced number of layers
num_decoder_layers = 12  # Reduced number of layers
dim_feedforward = 4096  # Reduced feedforward dimension

model = MokiTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward).to(device)

# Mixed precision training setup
scaler = GradScaler()

# Training loop
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
criterion = nn.CrossEntropyLoss()

# Gradient clipping value
max_grad_norm = 0.5

# Model checkpointing
checkpoint_path = 'moki/moki_checkpoint.pth'
best_loss = float('inf')

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_idx, src in enumerate(dataloader):
        src = src.to(device)
        tgt = src[:, :-1].to(device)
        tgt_y = src[:, 1:].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(src, tgt)
            loss = criterion(output.view(-1, vocab_size), tgt_y.view(-1))
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

        if batch_idx % 10 == 0:  # Adjusted logging frequency
            print(f"Epoch [{epoch+1}/{10}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    scheduler.step()
    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {average_loss:.4f}")

    # Save model checkpoint if it improves
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at Epoch {epoch+1} with Average Loss: {average_loss:.4f}")

print("Training complete!")
