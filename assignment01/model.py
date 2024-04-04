import torch.nn as nn
from torchvision import models

class BirdClassifier(nn.Module):
    """
    This model incorporates transfer learning 
    and additional regularization for improved performance on fine-grained image
    classification tasks.
    
    Parameters:
        num_classes (int): The number of classes in the dataset. Default is 200 for CUB-200.
        fine_tune_start (int): The layer index from which to start fine-tuning the model.
                               Set to a negative value to fine-tune all layers. Default is 5.

    Example usage:
        >>> model = BirdClassifier(num_classes=200, fine_tune_start=5)
    """
    
    def __init__(self, num_classes=200, fine_tune_start=5):
        super(BirdClassifier, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Fine-tuning configuration: layers before `fine_tune_start` are frozen,
        # others are left for fine-tuning based on the new dataset.
        if fine_tune_start < 0:
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            # Freeze all layers up to `fine_tune_start`
            layers = list(self.base_model.children())[:fine_tune_start]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Replace the original fully connected layer with a new one that includes
        # dropout for regularization and outputs to the number of classes in the dataset.
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters:
            x (torch.Tensor): Input batch of images (tensor).

        Returns:
            torch.Tensor: The model's predictions for the input batch.
        """
        # Pass the input through the base model and the modified fully connected layers.
        return self.base_model(x)
