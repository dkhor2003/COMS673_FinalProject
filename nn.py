import torch
import torch.nn as nn
import torch.optim as optim

class FalloverPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(FalloverPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return self.sigmoid(out).squeeze(-1)

# Example usage
num_features = 18  # Adjust according to your data
window_size = 50   # Adjust according to your data
model = FalloverPredictor(input_size=num_features)

# Dummy input (batch_size, window_size, num_features)
dummy_input = torch.randn(32, window_size, num_features)
out = model(dummy_input)
print(out.shape)  # Should be (32,)
