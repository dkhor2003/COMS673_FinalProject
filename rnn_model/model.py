import torch
import torch.nn as nn

class RNNFallPredictor(nn.Module):
    def __init__(self, input_size=54, hidden_size=64, num_layers=2, dropout=0.3, rnn_type="rnn"):
        super().__init__()
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("rnn_type must be 'lstm' or 'rnn'")

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # use last time step
        return self.fc(out)
