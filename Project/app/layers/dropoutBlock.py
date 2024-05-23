import torch
import torch.nn as nn

class DropoutBlock(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)
