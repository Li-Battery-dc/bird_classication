from sklearn import svm
import numpy as np

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3):
        self.kernel = kernel
        self.C = C
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=42)

    def train(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def evaluate(self, features, labels):

        predictions = self.predict(features)
        correct = np.sum(predictions == labels)
        total = len(labels)
        accuracy = correct / total
        return accuracy
    
