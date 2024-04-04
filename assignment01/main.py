from dataset import DatasetPreprocessing, create_label_map

DEBUG = True

label_map = create_label_map(file_path='data/train.txt')
dataset = DatasetPreprocessing(dataset_path='data/images', 
                                annotation_file='data/train.txt', 
                                label_map=label_map, 
                                transform=None, 
                                debug=DEBUG)