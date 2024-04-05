from ..model import BirdClassifier

model = BirdClassifier(num_classes=200, fine_tune_start=5)
print(model)