# %%
import numpy as np
import pandas as pd
import cvxopt
from cvxopt import matrix, solvers
import os
import cv2

# %%
class_paths = ['./svm/train/0', './svm/train/1', './svm/train/2', 
               './svm/train/3', './svm/train/4', './svm/train/5']
images = [[] for i in range(6)]
for i  in range(len(class_paths)):
    for filename in os.listdir(class_paths[i]):
        img = cv2.imread(os.path.join(class_paths[i], filename))
        if img is not None:
            resized_img = cv2.resize(img, (16, 16))
            normalized_img = resized_img / 255.0
            flattened_img = normalized_img.flatten()
            images[i].append(flattened_img)

# %%
validation_paths = ['./svm/val/0', './svm/val/1', './svm/val/2', 
                    './svm/val/3', './svm/val/4', './svm/val/5']
validation_images = [[] for i in range(6)]
for i  in range(len(validation_paths)):
    for filename in os.listdir(validation_paths[i]):
        img = cv2.imread(os.path.join(validation_paths[i], filename))
        if img is not None:
            resized_img = cv2.resize(img, (16, 16))
            normalized_img = resized_img / 255.0
            flattened_img = normalized_img.flatten()
            validation_images[i].append(flattened_img)

# %%
def gaussian_matrix(X, gamma=0.001):
    squared_distances = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, X.T) + np.sum(X**2, axis=1, keepdims=True).T
    kernel_matrix = np.exp(-gamma * squared_distances)
    return kernel_matrix

# %%
class Gaussian_SVM:

    def __init__(self, X, Y, C, gamma):
        self.C = C
        self.gamma = gamma
        self.X = X
        self.Y = Y

    def train(self):
        X = self.X
        Y = self.Y
        self.m = X.shape[0]
        self.n = X[0].shape[0]
        H = gaussian_matrix(X, gamma=self.gamma)
        P = matrix(H * (Y[:, np.newaxis] * Y), tc='d')
        q = matrix(-np.ones((self.m, 1)), tc='d')
        G = matrix(np.vstack((np.eye(self.m)*-1,np.eye(self.m))), tc='d')
        h = matrix(np.hstack((np.zeros(self.m), np.ones(self.m) * 1)), tc='d')
        A = matrix(Y.reshape(1, -1), tc='d')
        b = matrix(np.zeros(1), tc='d')
        sol = solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol['x'])
        self.num_sv = np.sum(self.alphas > 1e-5)
        self.support_vectors = []
        self.y_sv = []
        self.alpha_sv = []
        for i in range(self.m):
            if(self.alphas[i] > 1e-5):
                self.support_vectors.append(X[i])
                self.alpha_sv.append(self.alphas[i])
                self.y_sv.append(Y[i])
        self.support_vectors = np.array(self.support_vectors)
        self.alpha_sv = np.array(self.alpha_sv)
        self.y_sv = np.array(self.y_sv)
        kernel_matrix_bias = gaussian_matrix(self.support_vectors)
        new_matrix = kernel_matrix_bias * self.alpha_sv.reshape(-1, 1) * self.y_sv.reshape(-1, 1)
        bias_list = self.y_sv - np.sum(new_matrix, axis=0)
        self.b = 0
        num = 0
        for i in range(len(bias_list)):
            if(self.alpha_sv[i] < 1e-5 or (1 - self.alpha_sv[i]) < 1e-5): continue
            self.b += bias_list[i]
            num += 1
        self.b /= num
        self.b

    def prediction(self, x):
        prediction =  np.sum(np.exp(-0.001*np.sum((self.support_vectors - x)**2, axis = 1)) * self.y_sv * self.alpha_sv.reshape(self.alpha_sv.shape[0], ))  + self.b
        return prediction
    
    def predict_class(self, x):
        prediction = self.prediction(x)
        return 1 if prediction>=0 else -1

# %%
multi_class_svms = [[None for j in range(6)] for i in range(6)]
for i in range(6):
    for j in range(i + 1, 6):
        print(f"Starting SVM for classes {i}, {j}")
        X = np.vstack((np.array(images[i]), np.array(images[j])))
        Y = np.hstack((np.ones(len(images[i])), -np.ones(len(images[j]))))
        svm = Gaussian_SVM(X, Y, 1, 0.001)
        svm.train()
        multi_class_svms[i][j] = svm
        print(f"SVM trained for classes {i}, {j}")

# %%
def multiclass_predict(x):
    distances = np.zeros((6,))
    votes = np.zeros((6,))
    for i in range(len(multi_class_svms)):
        for j in range(len(multi_class_svms)):
            if(multi_class_svms[i][j] is None): continue
            distance = multi_class_svms[i][j].prediction(x)
            if(distance >= 0): 
                votes[i] += 1
                distances[i] += abs(distance)
            else: 
                votes[j] += 1
                distances[j] += abs(distance)
    max_vote_index = np.argmax(votes)
    equal_vote_indices = np.where(votes == votes[max_vote_index])[0]
    if(len(equal_vote_indices) == 1): return max_vote_index
    else:
        max_distance_index = equal_vote_indices[np.argmax(distances[equal_vote_indices])]
        return max_distance_index

# %%
valid_correct = 0
valid_incorrect = 0
misclassified = []
misclassified_class = [[] for i in range(6)]
misclassified_prediction = [[] for i in range(6)]
confusion_matrix = np.zeros((6, 6))
for i in range(len(validation_images)):
    print(f"Starting validation for class {i}")
    j = 0
    for image in validation_images[i]:
        print(j, end=' ')
        prediction = multiclass_predict(image)
        confusion_matrix[i][prediction] += 1
        if(prediction == i): valid_correct += 1
        else: 
            valid_incorrect += 1
            misclassified.append(img)
            misclassified_class[i].append(j)
            misclassified_prediction[i].append(prediction)
        j += 1
    print()

# %%
print(f"Correct: {valid_correct}\nIncorrect: {valid_incorrect}\nAccuracy: {valid_correct / (valid_correct + valid_incorrect)}")
print(f"Confusion Matrix:\n{confusion_matrix}")

# %%
actual = 1
labelled = 5
directory_images = os.listdir(f"./svm/val/{actual}/")
misclassed = []
for i in range(len(misclassified_class[actual])):
    if(misclassified_prediction[actual][i] == labelled): misclassed.append(directory_images[misclassified_class[actual][i]])


