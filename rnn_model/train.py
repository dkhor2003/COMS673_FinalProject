import torch
import torch.nn as nn
import torch.optim as optim
from model import RNNFallPredictor
from dataloader import get_dataloaders
from tqdm import tqdm
import argparse

# Argument parser for terminal input
parser = argparse.ArgumentParser(description="Train an RNN or LSTM fall predictor.")
parser.add_argument('--rnn_type', choices=['rnn', 'lstm'], default='rnn', help="Choose RNN type: 'rnn' or 'lstm'")
args = parser.parse_args()

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
SEQ_LEN = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*50)
print(f"[CONFIG] RNN Type     : {args.rnn_type}")
print(f"[CONFIG] Epochs       : {EPOCHS}")
print(f"[CONFIG] Batch Size   : {BATCH_SIZE}")
print(f"[CONFIG] Seq Length   : {SEQ_LEN}")
print(f"[CONFIG] Learning Rate: {LR}")
print("="*50)

# Data path
DATA_DIR = "data/g1_traj_1"

# Load data
train_loader, val_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

# Initialize model
model = RNNFallPredictor(input_size=54, rnn_type=args.rnn_type)
model.to(DEVICE)

print(f"[INFO] Using RNN type: {args.rnn_type.upper()}")

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.sigmoid(outputs).round()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Accuracy={acc:.2f}%")

# Final report
print("\n" + "="*50)
print("🏁 Training Complete")
print(f"🧠 RNN Type             : {args.rnn_type.upper()}")
print(f"📉 Final Epoch Loss     : {running_loss/len(train_loader):.4f}")
print(f"✅ Final Epoch Accuracy : {acc:.2f}%")
print("="*50)

# Save model
torch.save(model.state_dict(), "rnn_fall_model.pt")
