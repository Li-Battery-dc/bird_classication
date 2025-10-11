from svm.Classifier import SVMClassifier
from dataloader.Dataloader import DataLoader
import random

def svm_method(data_root='/home/stu12/homework/MLPR/data'):
    
    train_loader = DataLoader(data_root, split='train', mode='feature')
    classifier = SVMClassifier(kernel='linear', C=1.0)

    random.seed(42)
    target_labels = random.sample(range(1, len(train_loader.class_dirs)+1), 10)
    train_features, train_labels = train_loader.get_data_and_labels(target_labels=target_labels)

    classifier.train(train_features, train_labels)

    val_loader = DataLoader(data_root, split='val', mode='feature')
    val_features, val_labels = val_loader.get_data_and_labels(target_labels=target_labels)
    val_accuracy = classifier.evaluate(val_features, val_labels)
    print(f"验证集准确率: {val_accuracy * 100:.2f}%")
