import torch
import torch.nn as nn

class NormalizationBlock(nn.Module):
    def __init__(self, num_features):
        super(NormalizationBlock, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)