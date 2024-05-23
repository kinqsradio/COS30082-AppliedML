import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        return hn[-1] 
