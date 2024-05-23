import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

log_dir = 'runs/training_logs'

def calculate_similarity(embedding1, embedding2, method='euclidean'):
    if method == 'euclidean':
        return -torch.cdist(embedding1.unsqueeze(0), embedding2.unsqueeze(0))  # Use negative distance for similarity
    elif method == 'cosine':
        return F.cosine_similarity(embedding1, embedding2)
    else:
        raise ValueError("Unsupported similarity method")

def plot_roc_curve(fpr, tpr, roc_auc, epoch):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Epoch {epoch + 1}')
    plt.legend(loc='lower right')
    plt.savefig(f'{log_dir}/roc_curve_epoch_{epoch + 1}.png')
    plt.close()

def evaluate_verification(model, val_loader, device, epoch):
    similarities = []
    labels = []

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        embeddings = model.extract_features(inputs, aggregate=True)
        
        for i in range(embeddings.size(0)):
            for j in range(i + 1, embeddings.size(0)):
                similarity = calculate_similarity(embeddings[i], embeddings[j], method='euclidean')
                label = 1 if targets[i] == targets[j] else 0
                similarities.append(similarity.item())
                labels.append(label)

    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)
    
    plot_roc_curve(fpr, tpr, roc_auc, epoch)

    return roc_auc
