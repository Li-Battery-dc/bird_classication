from cvxopt import matrix, solvers
import numpy as np

class SVMClassifier:
    '''
    One vs One 多分类SVM的基础实现
    '''
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, epsilon=1e-5):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.epsilon = epsilon
        
        self.classifiers = {}
        self.classes = None

    def train(self, features, labels):
        
        self.classes = np.unique(labels)
        n_classes = len(self.classes)
        
        # one vs one 训练多个二分类SVM
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i = self.classes[i]
                class_j = self.classes[j]
                
                # 提取当前类别对的数据
                mask = (labels == class_i) | (labels == class_j)
                X = features[mask]
                Y = labels[mask]
                
                # 训练二分类SVM
                binary_svm = BinarySVM(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, epsilon=self.epsilon)
                binary_svm.train(X, Y)

                self.classifiers[(class_i, class_j)] = binary_svm

    def predict(self, features):
        n_samples = features.shape[0]
        n_classes = len(self.classes)
        
        votes = np.zeros((n_samples, n_classes))

        for (class_i, class_j), classifier in self.classifiers.items():
            predictions = classifier.predict(features)
            for idx, pred in enumerate(predictions):
                if pred == class_i:
                    votes[idx, np.where(self.classes == class_i)[0][0]] += 1
                else:
                    votes[idx, np.where(self.classes == class_j)[0][0]] += 1

        # 预测结果为得票最多的类别
        return self.classes[np.argmax(votes, axis=1)]

    def evaluate(self, features, labels):

        predictions = self.predict(features)
        correct = np.sum(predictions == labels)
        total = len(labels)
        accuracy = correct / total
        return accuracy
    
class BinarySVM:
    """
    二分类SVM的基础实现
    """
    
    def __init__(self, kernel='linear', C=1.0, gamma='scale', degree=3, epsilon=1e-5):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.epsilon = epsilon
        
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.unique_labels = None
    
    def _kernel_function(self, X1, X2):
        """计算核函数矩阵"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'poly':
            if self.gamma == 'scale':
                gamma_val = 1.0 / X1.shape[1]
            else:
                gamma_val = self.gamma
            return (gamma_val * np.dot(X1, X2.T) + 1) ** self.degree
        
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma_val = 1.0 / X1.shape[1]
            else:
                gamma_val = self.gamma

            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
            dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma_val * dists)
        
        else:
            raise ValueError(f"不支持的核函数: {self.kernel}")
    
    def train(self, features, labels):
        """训练二分类SVM"""
        self.unique_labels = np.unique(labels)
        if len(self.unique_labels) != 2:
            raise ValueError("标签必须是二分类的")
        # 做了一次映射，后续predict映射回原标签
        labels = np.where(labels == self.unique_labels[0], -1, 1)

        n_samples = features.shape[0]
        
        self.X_train = features
        self.y_train = labels
        
        # 计算核矩阵
        K = self._kernel_function(features, features)
        
        # 构建二次规划问题
        # cvxopt: 
        # minimize: (1/2)a^T Pa + q^T a
        # subject to: Ga ≤ h
        #             Aa = b
        # max L(α) = Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        # min (1/2)a^T K a - a
        # subject to: αᵢ ≤ C
        #             -αᵢ ≤ 0
        #             Σᵢ αᵢyᵢ = 0
        P = matrix(np.outer(labels, labels) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(labels, (1, n_samples))
        b = matrix(0.0)

        # 求解二次规划问题
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)

        # 提取拉格朗日乘子
        self.alpha = np.ravel(solution['x'])

        # 支持向量
        sv_mask = self.alpha > self.epsilon # 只取大于0的αᵢ
        self.support_vectors = self.X_train[sv_mask]
        self.support_vector_labels = self.y_train[sv_mask]
        self.alpha = self.alpha[sv_mask]

        # 计算偏置项b
        # 利用支持向量边界约束条件计算b
        margin_sv_mask = (self.alpha > self.epsilon) & (self.alpha < self.C - self.epsilon) # 排除软间隔支持向量
        if np.any(margin_sv_mask):
            K_sv = self._kernel_function(self.support_vectors, self.support_vectors)
            # b = yᵢ - Σ(αⱼyⱼK(xⱼ,xᵢ))
            self.b = np.mean(self.support_vector_labels[margin_sv_mask] - 
                             np.sum(self.alpha[:, np.newaxis] * self.support_vector_labels[:, np.newaxis] * K_sv[:, margin_sv_mask], axis=0))
        else:
            # 极端情况，没有边界支持向量，使用所有支持向量计算b
            K_sv = self._kernel_function(self.support_vectors, self.support_vectors)
            self.b = np.mean(self.support_vector_labels - 
                           np.sum(self.alpha[:, np.newaxis] * self.support_vector_labels[:, np.newaxis] * K_sv, axis=0))
        


    def decision_func(self, features):
        """计算决策函数值"""
        K = self._kernel_function(self.X_train, features)
        decision = np.dot(self.alpha * self.support_vector_labels, K) + self.b
        return decision
    
    def predict(self, features):
        """预测标签"""
        decision = self.decision_func(features)
        bi_predict = np.sign(decision)

        predict_labels = np.where(bi_predict == -1, self.unique_labels[0], self.unique_labels[1]) # 映射回原标签
        return predict_labels