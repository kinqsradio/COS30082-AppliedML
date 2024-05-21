from torch.utils.data import DataLoader
from dataset import DatasetPreprocessing, create_label_map
from model import BirdClassifier
from helper import train_model

if __name__ == '__main__':

    label_map = create_label_map(file_path='data/train.txt')
    train_dataset = DatasetPreprocessing(dataset_path='data/train', annotation_file='data/train.txt', debug=False, label_map=label_map)
    test_dataset = DatasetPreprocessing(dataset_path='data/test', annotation_file='data/test.txt', debug=False, label_map=label_map)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model
    model = BirdClassifier(num_classes=200, fine_tune_start=5)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20)