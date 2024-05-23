import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import torch.nn as nn

BASE_PATH = './dataset/classification_data'
TRAIN_PATH = os.path.join(BASE_PATH, 'train_data')  
VAL_PATH = os.path.join(BASE_PATH, 'val_data')  
TEST_PATH = os.path.join(BASE_PATH, 'test_data')



class DataPreprocessing:
    def __init__(self, root, image_size=(224, 224), load_all=False):
        """
        Initialize the data preprocessing module.
        Args:
            root (str): The root directory containing the image data.
            image_size (tuple): The target size of images (width, height).
            load_all (bool): If True, loads all images into memory at initialization.
        """
        self.root = root
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.unique_label_ids = os.listdir(self.root)
        print("Unique labels:", len(self.unique_label_ids))
        
        self.image_paths = []
        self.label_nums = []
        self.label_ids = []
        self.num_classes = len(self.unique_label_ids)
        self.unique_label_nums = list(range(self.num_classes))
        
        for label_num, label_id in enumerate(self.unique_label_ids):
            folder_path = os.path.join(self.root, label_id)
            images_list = os.listdir(folder_path)
            images_list_abs = [os.path.join(folder_path, image) for image in images_list]
            self.image_paths.extend(images_list_abs)
            num_images = len(images_list)
            self.label_ids.extend([label_id] * num_images)
            self.label_nums.extend([label_num] * num_images)
        
        self.data_len = len(self.label_nums)
        print("Total samples:", self.data_len)
        self.images = {}
        
        if load_all:
            print("Loading all images...")
            for i in tqdm(range(self.data_len)):
                self.images[i] = self.__load_image(self.image_paths[i])

    def __load_image(self, path):
        """ Load an image with specified transformations. """
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        """ Lazy load image. """
        if index in self.images:
            return self.images[index], self.label_nums[index]
        else:
            image = self.__load_image(self.image_paths[index])
            return image, self.label_nums[index]

    def reduce_samples(self, ids):
        """ Reduce the dataset to only include specified ids. 
        Args:
            ids (list): A list of label IDs that should be kept in the dataset.
        """
        current_len = self.data_len
        new_image_paths = []
        new_label_nums = []
        new_label_ids = []
        
        # Create a set for quick lookup
        id_set = set(ids)

        for i in range(current_len):
            label_id = self.label_ids[i]
            image_path = self.image_paths[i]
            
            if label_id in id_set:
                new_label_ids.append(label_id)
                new_image_paths.append(image_path)
                # Ensure the new label number is correctly mapped to the position in 'ids'
                new_label_nums.append(ids.index(label_id)) 
        
        # Update internal state to reflect the reduced data
        self.label_ids = new_label_ids
        self.image_paths = new_image_paths
        self.label_nums = new_label_nums
        self.unique_label_ids = ids
        self.num_classes = len(ids)
        self.unique_label_nums = list(range(self.num_classes))
        self.data_len = len(new_label_ids)
        
        # Clear the images dictionary to reset lazy loading mechanism
        self.images = dict()

        print("Before reduction total samples:", current_len)
        print("After reduction total samples:", self.data_len)
        print("Reduced unique labels count:", self.num_classes, "Label nums example:", self.label_nums[:10])
        print("New Unique labels example:", self.unique_label_nums[:10])

        
# Test Function
if __name__ == '__main__':
    params = dict(
        batch_size=128,
        num_labels=300,
    )
        
    print("Training Data")
    train_dataset = DataPreprocessing(TRAIN_PATH)
    labels = train_dataset.unique_label_ids[:params['num_labels']]
    train_dataset.reduce_samples(labels)
    
    print("\nValidation Data")
    val_dataset = DataPreprocessing(VAL_PATH)
    val_dataset.reduce_samples(labels)
    
    print("\nTest Data")
    test_dataset = DataPreprocessing(TEST_PATH)
    test_dataset.reduce_samples(labels)

    train_loader_args = dict(shuffle=True,
                            batch_size=params['batch_size'])
    
    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    val_loader = data.DataLoader(val_dataset, **train_loader_args)
    test_loader = data.DataLoader(val_dataset, **train_loader_args)

    NUM_CLASSES = train_dataset.num_classes
    print(f'Number of classes: {NUM_CLASSES}')