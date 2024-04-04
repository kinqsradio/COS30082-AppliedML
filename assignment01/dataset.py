import re
import os
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def create_label_map(file_path):
    label_map = {}
    with open(file_path, 'r') as file:
        for line in file:
            image_name, label_id = line.strip().split(' ')
            # Using regex to remove the trailing "_number" from image names to extract the bird class name
            class_name = re.sub(r'(_\d+)?\.jpg$', '', image_name)
            # Replace underscores with spaces to get the final class name
            class_name = class_name.replace('_', ' ')
            label_id = int(label_id)
            if label_id not in label_map:
                label_map[label_id] = class_name

    return {k: re.sub(r'\d+$', '', v).strip() for k, v in label_map.items()}

class DatasetPreprocessing(Dataset):
    """
    DatasetPreprocessing for loading and preprocessing Caltech-UCSD Birds 200 (CUB-200) images and annotations,
    with an optional debug mode for logging details about data loading and preprocessing steps.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        annotation_file (str): Path to the annotations file.
        transform (callable, optional): Transform to be applied on a sample for preprocessing.
        debug (bool, optional): If True, print debug information. Defaults to False.
    """
    def __init__(self, 
                 dataset_path: str, 
                 annotation_file: str, 
                 label_map: Dict[int, str],
                 transform: transforms.Compose = transforms.Compose([
                     transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels.
                     transforms.ToTensor(),          # Convert the image to a PyTorch tensor.
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the tensor image.
                                          std=[0.229, 0.224, 0.225])
                 ]),
                 debug: bool = False) -> None:
        self.dataset_path = dataset_path
        self.annotation_file = annotation_file
        self.label_map = label_map
        self.transform = transform
        self.debug = debug
        self.images, self.labels = self.load_dataset()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetches the preprocessed image and label at the specified index, with optional debug logging.
        
        Args:
            idx (int): Index of the item.
        
        Returns:
            Tuple[torch.Tensor, int]: The preprocessed image tensor and its label.
        """
        image_path = os.path.join(self.dataset_path, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        bird_name = self.label_map[label]
        
        if self.debug:
            print(f'Loading image: {image_path}, Label ID: {label}, Bird Name: {bird_name}')

            
        if self.transform is not None:
            if self.debug:
                print(f'Applying transform: {self.transform}')
            image = self.transform(image)
        
        return image, label

    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Loads dataset from the annotation file, with optional debug logging.
        
        Returns:
            Tuple[List[str], List[int]]: Lists of image filenames and their corresponding labels.
        """
        images, labels = [], []
        with open(self.annotation_file, 'r') as file:
            for line in file:
                image_name, label = line.strip().split(' ')
                images.append(image_name)
                labels.append(int(label))
                
        if self.debug:
            print(f'Total images loaded: {len(images)}')
                
        return images, labels

