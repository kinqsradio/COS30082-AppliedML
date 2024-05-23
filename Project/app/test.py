import torch
from models import FaceRecognition, CONFIG
from datasets import DataPreprocessing, TRAIN_PATH, VAL_PATH, TEST_PATH
from torch.utils.data import DataLoader


# Test the model
def main():
    
    # Initialize datasets using predefined paths
    train_dataset = DataPreprocessing(TRAIN_PATH)
    val_dataset = DataPreprocessing(VAL_PATH)
    test_dataset = DataPreprocessing(TEST_PATH)

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    model = FaceRecognition(CONFIG)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dummy forward pass to test integration
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        print(f'dummy input shape: {images.shape}')

        # Forward pass
        outputs = model(images)
        print("Output shape:", outputs.shape)
        break  # Just do one batch for testing

if __name__ == '__main__':
    main()