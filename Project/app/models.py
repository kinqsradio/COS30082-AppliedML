import torch
import torch.nn as nn
from layers.convBlock import ConvBlock
from layers.normalizationBlock import NormalizationBlock
from layers.dropoutBlock import DropoutBlock
from layers.lstmBlock import LSTMBlock
from layers.attentionBlock import AttentionBlock
from layers.residualBlock import ResidualBlock

CONFIG = [
        {'type': 'conv', 'params': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
        {'type': 'norm', 'params': {'num_features': 64}},
        {'type': 'residual', 'params': {'channels': 64}},  # First residual block
        {'type': 'dropout', 'params': {'p': 0.5}},

        {'type': 'conv', 'params': {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
        {'type': 'norm', 'params': {'num_features': 128}},
        {'type': 'residual', 'params': {'channels': 128}},  # Second residual block
        {'type': 'maxpool', 'params': {'kernel_size': 2, 'stride': 2, 'padding': 0}},  # Max pooling for dimensionality reduction

        {'type': 'conv', 'params': {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
        {'type': 'norm', 'params': {'num_features': 256}},
        {'type': 'residual', 'params': {'channels': 256}},  # Third residual block
        {'type': 'dropout', 'params': {'p': 0.3}},  # Increased dropout

        {'type': 'conv', 'params': {'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}},
        {'type': 'norm', 'params': {'num_features': 512}},
        {'type': 'residual', 'params': {'channels': 512}},  # Fourth residual block
        {'type': 'avgpool', 'params': {'kernel_size': 2, 'stride': 2, 'padding': 0}},  # Average pooling for variation

        {'type': 'adaptive_pool', 'params': {'output_size': (1, 1)}},  # Global adaptive pooling
        {'type': 'dropout', 'params': {'p': 0.3}},  # Additional dropout before sequence processing

        {'type': 'lstm', 'params': {'input_size': 512, 'hidden_size': 512, 'num_layers': 2, 'batch_first': True}},  # Adjusted input_size for LSTM
        {'type': 'attention', 'params': {'in_features': 512}},  # Attention mechanism
        {'type': 'dropout', 'params': {'p': 0.2}},  # Final dropout layer

        {'type': 'fc', 'params': {'in_features': 512, 'out_features': 256}},  # Fully connected layer
        {'type': 'dropout', 'params': {'p': 0.2}},  # Dropout after FC layer
        {'type': 'fc', 'params': {'in_features': 256, 'out_features': 300}}  # Final fully connected layer for classification (300 classes)
    ]

class FaceRecognition(nn.Module):
    def __init__(self, config):
        super(FaceRecognition, self).__init__()
        self.layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()  # Separate list for fully connected layers
        self.anti_spoofing_layers = nn.ModuleList()  # Layers for the anti-spoofing module

        for layer_conf in config:
            layer_type = layer_conf['type']
            if layer_type == 'conv':
                self.layers.append(ConvBlock(**layer_conf['params']))
            elif layer_type == 'norm':
                self.layers.append(NormalizationBlock(**layer_conf['params']))
            elif layer_type == 'dropout':
                self.layers.append(DropoutBlock(**layer_conf['params']))
            elif layer_type == 'residual':
                self.layers.append(ResidualBlock(**layer_conf['params']))
            elif layer_type == 'attention':
                self.attention_config = layer_conf['params']
            elif layer_type == 'fc':
                self.fc_layers.append(nn.Linear(**layer_conf['params']))
            elif layer_type == 'maxpool':
                self.layers.append(nn.MaxPool2d(**layer_conf['params']))
            elif layer_type == 'avgpool':
                self.layers.append(nn.AvgPool2d(**layer_conf['params']))
            elif layer_type == 'adaptive_pool':
                self.layers.append(nn.AdaptiveAvgPool2d(**layer_conf['params']))

        self.lstm = LSTMBlock(input_size=512, hidden_size=512, num_layers=2, batch_first=True) if any(l['type'] == 'lstm' for l in config) else None
        self.attention_block = None if not any(l['type'] == 'attention' for l in config) else AttentionBlock(**self.attention_config)
        
        # Anti-spoofing module configuration
        self.anti_spoofing_layers.append(ConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1))
        self.anti_spoofing_layers.append(NormalizationBlock(num_features=32))
        self.anti_spoofing_layers.append(DropoutBlock(p=0.3))
        self.anti_spoofing_layers.append(ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        self.anti_spoofing_layers.append(NormalizationBlock(num_features=64))
        self.anti_spoofing_layers.append(DropoutBlock(p=0.3))
        self.anti_spoofing_fc = nn.Linear(64 * 112 * 112, 1)  # 224x224 image size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # print(f"After {layer.__class__.__name__}, shape: {x.shape}")/

        x = x.view(x.size(0), -1)  # [batch, features]
        # print(f"Flattened to: {x.shape}")

        if self.lstm:
            x = x.unsqueeze(1)  # [batch, seq_len, features]
            x = self.lstm(x)

        if self.attention_block:
            x = self.attention_block(x)

        for fc in self.fc_layers:
            x = fc(x)
            # print(f"After {fc.__class__.__name__}, shape: {x.shape}")

        return x

    def extract_features(self, x, stop_at_layer=None, aggregate=False):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if aggregate:
                features.append(x.view(x.size(0), -1)) 
            if stop_at_layer is not None and i == stop_at_layer:
                break
        if aggregate:
            x = torch.cat(features, dim=1)
        else:
            x = x.view(x.size(0), -1)  # [batch, features]
        return x

    def anti_spoofing(self, x):
        for layer in self.anti_spoofing_layers:
            x = layer(x)
        x = x.view(x.size(0), -1) 
        x = self.anti_spoofing_fc(x)
        return torch.sigmoid(x) 



# Test the model
if __name__ == '__main__':
    from helper import calculate_similarity
    model = FaceRecognition(CONFIG)
    dummy_input1 = torch.randn(1, 3, 224, 224)
    dummy_input2 = torch.randn(1, 3, 224, 224)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input1 = dummy_input1.to(device)
    dummy_input2 = dummy_input2.to(device)

    try:
        # For classification
        output = model(dummy_input1)
        print("Output shape (classification):", output.shape)

        # For verification
        embedding1 = model.extract_features(dummy_input1, aggregate=True)
        embedding2 = model.extract_features(dummy_input2, aggregate=True)  # Using a different dummy image
        similarity = calculate_similarity(embedding1, embedding2, method='euclidean')
        print("Similarity (verification):", similarity)

        # For anti-spoofing
        spoof_prob = model.anti_spoofing(dummy_input1)
        print("Spoof probability:", spoof_prob)

        print("Test Passed: Output generated successfully.")
    except Exception as e:
        print("Test Failed:", e)
