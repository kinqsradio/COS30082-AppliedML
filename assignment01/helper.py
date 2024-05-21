import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import pandas as pd
from io import BytesIO
import requests


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = 'models'
        model_save_path = os.path.join(path, f'checkpoint.pt')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        self.val_loss_min = val_loss
        
        
def train_step(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    loss = running_loss / len(loader.dataset)
    accuracy = running_corrects.double() / len(loader.dataset)
    return loss, accuracy

def val_step(model, loader, device, criterion):
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
    loss = val_loss / len(loader.dataset)
    accuracy = val_corrects.double() / len(loader.dataset)
    return loss, accuracy

def train_model(model, train_loader, val_loader, num_epochs=25):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)

    # Initialize a list to store the logs, which will be converted to DataFrame later
    logs = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = val_step(model, val_loader, device, criterion)

        # Log the current epoch data using dictionary and add it to the list
        logs.append({
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train Accuracy': train_acc,
            'Validation Loss': val_loss,
            'Validation Accuracy': val_acc
        })

        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Validation - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Convert the list of logs into a DataFrame and export to CSV
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv('training_metrics.csv', index=False)
        
    print('Training complete and logs have been saved to training_metrics.csv')
    return model

def evaluate_model(model, test_loader, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # Set the model to evaluation mode
    correct_preds = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total += labels.size(0)
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (preds[i] == label).item()
                class_total[label] += 1

    top_1_accuracy = correct_preds / total
    average_accuracy_per_class = sum([class_correct[i] / class_total[i] for i in range(num_classes) if class_total[i] != 0]) / num_classes
    
    print(f'Top-1 Accuracy: {top_1_accuracy:.4f}')
    print(f'Average Accuracy Per Class: {average_accuracy_per_class:.4f}')
    
    return {
        'top_1_accuracy': top_1_accuracy,
        'average_accuracy_per_class': average_accuracy_per_class
    }

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model, image_path, label_map):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f'predicted: {predicted}')
        predicted_class_index = predicted.item()
        print(f'predicted_class_index: {predicted_class_index}')
        predicted_class_name = label_map[predicted_class_index]
        print(f'predicted_class_name: {predicted_class_name}')
    return predicted_class_name


def predict_online(model, url, label_map):
    response = requests.get(url)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(BytesIO(response.content))
    image = transform(img).unsqueeze(0)  # Add batch dimension

    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = label_map[predicted_class_index]
        
    return predicted_class_name, img
    