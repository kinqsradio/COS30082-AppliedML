import os
import json
from dataset import DatasetPreprocessing, create_label_map

label_map = create_label_map(file_path='data/train.txt')
print(json.dumps(label_map, indent=4))
dataset = DatasetPreprocessing(dataset_path='data/train', annotation_file='data/train.txt', debug=True, label_map=label_map)
for i in range(5):
    image, label = dataset[i]
