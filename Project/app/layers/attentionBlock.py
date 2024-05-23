import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features // 8)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features // 8, 1)

    def forward(self, x):
        # Assuming x is pooled to (batch_size, features) shape
        scores = self.fc1(x)
        scores = self.relu(scores)
        scores = self.fc2(scores)
        alpha = torch.softmax(scores, dim=0)
        context = torch.sum(alpha * x, dim=0, keepdim=True)
        return context