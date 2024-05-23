import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import FaceRecognition, CONFIG
from datasets import DataPreprocessing, TRAIN_PATH, VAL_PATH, TEST_PATH
from helper import evaluate_verification
import os

# tensorboard --logdir=runs/training_logs
log_dir = 'runs/training_logs'
model_save_path = 'models/train/best_model/'
best_model_path = os.path.join(model_save_path, 'best_model.pt')
best_val_model_path = os.path.join(model_save_path, 'best_val_model.pt')

# Ensure model save directory exists
os.makedirs(model_save_path, exist_ok=True)

def train(num_epochs=25, warmup_episodes=50):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize datasets using predefined paths
    train_dataset = DataPreprocessing(TRAIN_PATH)
    val_dataset = DataPreprocessing(VAL_PATH)
    test_dataset = DataPreprocessing(TEST_PATH)

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model initialization
    model = FaceRecognition(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    best_val_acc = 0.0
    best_model_acc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Save the best validation model
        if epoch >= warmup_episodes and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_val_model_path)
            print(f'Best validation model saved at epoch {epoch + 1}')

        # Save the best model based on training accuracy
        train_corrects = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                train_corrects += torch.sum(preds == labels.data)
        
        train_acc = train_corrects.double() / len(train_loader.dataset)
        if epoch >= warmup_episodes and train_acc > best_model_acc:
            best_model_acc = train_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved at epoch {epoch + 1}')

        # Verification evaluation
        roc_auc = evaluate_verification(model, val_loader, device, epoch)
        print(f'Epoch {epoch + 1}, AUC: {roc_auc:.4f}')
        writer.add_scalar('AUC/val', roc_auc, epoch)

    # Testing loop
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    writer.add_scalar('Accuracy/test', test_acc)

    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    train(num_epochs=10, warmup_episodes=2)
