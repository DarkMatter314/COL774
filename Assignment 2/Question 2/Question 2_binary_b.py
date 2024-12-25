# %%
import numpy as np
import pandas as pd
import cvxopt
from cvxopt import matrix, solvers
import os
import cv2

# %%
class_3_path = './svm/train/3/'
image_3_list = []
for filename in os.listdir(class_3_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(class_3_path,filename))
        if img is not None:
            resized_img = cv2.resize(img, (16, 16))
            normalized_img = resized_img / 255.0
            flattened_img = normalized_img.flatten()
            image_3_list.append(flattened_img)

class_4_path = './svm/train/4/'
image_4_list = []
for filename in os.listdir(class_4_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(class_4_path,filename))
        if img is not None:
            resized_img = cv2.resize(img, (16, 16))
            normalized_img = resized_img / 255.0
            flattened_img = normalized_img.flatten()
            image_4_list.append(flattened_img)

# %%
m = len(image_3_list) + len(image_4_list)
n = image_4_list[0].shape[0]
m, n

# %%
X = np.array(image_3_list + image_4_list)
Y = np.array([-1] * len(image_3_list) + [1] * len(image_4_list))

# %%
def gaussian_matrix(X, gamma=0.001):
    squared_distances = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, X.T) + np.sum(X**2, axis=1, keepdims=True).T
    kernel_matrix = np.exp(-gamma * squared_distances)
    return kernel_matrix

# %%
def gaussian_kernel(x, y, gamma=0.001):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

# %%
H = gaussian_matrix(X)

# %%
Y.reshape(-1, 1).T.shape

# %%
PP = H * (Y[:, np.newaxis] * Y)

# %%
P = matrix(PP, tc='d')
q = matrix(-np.ones((m, 1)), tc='d')
G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))), tc='d')
h = matrix(np.hstack((np.zeros(m), np.ones(m) * 1)), tc='d')
A = matrix(Y.reshape(1, -1), tc='d')
b = matrix(np.zeros(1), tc='d')

# %%
import random
for i in range(10):
    one = random.randint(0, 4000)
    for j in range(10):
       two = random.randint(0, 4000)
       if(abs(H[one, two] - gaussian_kernel(X[one], X[two])) > 1e-15):
           print(f"Error at {one}, {two}, H = {H[one, two]}, K = {gaussian_kernel(X[one], X[two])}")

# %%
sol = solvers.qp(P, q, G, h, A, b)

# %%
alphas = np.array(sol['x'])

# %%
num_sv = len(alphas[alphas > 1e-5])
num_sv

# %%
support_vectors = []
y_sv = []
alpha_sv = []
for i in range(m):
    if(alphas[i] > 1e-5):
        support_vectors.append(X[i])
        alpha_sv.append(alphas[i])
        y_sv.append(Y[i])
support_vectors = np.array(support_vectors)
alpha_sv = np.array(alpha_sv)
y_sv = np.array(y_sv)

# %%
# b_list = []
# for i in range(m):
#     if(alphas[i] < 1e-5 or (1 - alphas[i]) < 1e-5): continue
kernel_matrix_bias = gaussian_matrix(support_vectors)
new_matrix = kernel_matrix_bias * alpha_sv.reshape(-1, 1) * y_sv.reshape(-1, 1)
bias_list = y_sv - np.sum(new_matrix, axis=0)
b = 0
num = 0
for i in range(len(bias_list)):
    if(alpha_sv[i] < 1e-5 or (1 - alpha_sv[i]) < 1e-5): continue
    b += bias_list[i]
    num += 1
b /= num
b

# %%
def predict(x, support_vectors, b, gamma = 0.001):
    prediction =  np.sum(np.exp(-0.001*np.sum((support_vectors - x)**2, axis = 1)) * y_sv * alpha_sv.reshape(alpha_sv.shape[0], ))  + b
    return 1 if prediction >= 0 else -1

# %%
validation_3_path = './svm/val/3/'
validation_image_3_list = []
for filename in os.listdir(validation_3_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(validation_3_path,filename))
        if img is not None:
            resized_img = cv2.resize(img, (16, 16))
            normalized_img = resized_img / 255.0
            flattened_img = normalized_img.flatten()
            validation_image_3_list.append(flattened_img)
validation_4_path = './svm/val/4/'
validation_image_4_list = []
for filename in os.listdir(validation_4_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(validation_4_path,filename))
        if img is not None:
            resized_img = cv2.resize(img, (16, 16))
            normalized_img = resized_img / 255.0
            flattened_img = normalized_img.flatten()
            validation_image_4_list.append(flattened_img)

# %%
correct = 0
incorrect = 0
confusion_matrix = [[0, 0], [0, 0]]
for img in validation_image_3_list:
    prediction = predict(img, support_vectors, b)
    if(prediction == -1):
        correct += 1
        confusion_matrix[0][0] += 1
    else:
        incorrect += 1
        confusion_matrix[0][1] += 1
for img in validation_image_4_list:
    prediction = predict(img, support_vectors, b)
    if(prediction == 1):
        correct += 1
        confusion_matrix[1][1] += 1
    else:
        incorrect += 1
        confusion_matrix[1][0] += 1
print(f"Correct: {correct} \nIncorrect: {incorrect} \nAccuracy: {correct / (correct + incorrect)}")
print(f"Confusion Matrix: \n{confusion_matrix}")

# %%
train_incorrect = 0
for img in image_3_list:
    prediction = predict(img, support_vectors, b)
    if(prediction != -1):
        train_incorrect += 1
for img in image_4_list:
    prediction = predict(img, support_vectors, b)
    if(prediction != 1):
        train_incorrect += 1
print(f"Correct: {m - train_incorrect}\nIncorrect: {train_incorrect}\nAccuracy: {(m - train_incorrect) / m}")

# %%
sorted_indices = np.argsort(alphas, axis=0)[::-1]
top_6 = sorted_indices[:6]

# %%
top_6_support_vectors = []
image_3_names = os.listdir(class_3_path)
image_4_names = os.listdir(class_4_path)
image_3_sv = []
image_4_sv = []
for i in top_6:
    if i[0] < len(image_3_names):
        top_6_support_vectors.append(image_3_names[i[0]])
        image_3_sv.append(image_3_names[i[0]])
    else:
        top_6_support_vectors.append(image_4_names[i[0] - len(image_3_names)])
        image_4_sv.append(image_4_names[i[0] - len(image_3_names)])

# %%
from PIL import Image
output_dir = './part_b_sv/'
i = 0
for img_name in image_3_sv:
    img = cv2.imread(os.path.join(class_3_path, img_name))
    if img is not None:
        resized_img = cv2.resize(img, (16, 16))
        pillow_image_resized = Image.fromarray(np.uint8(resized_img))
        pillow_image_original = Image.fromarray(np.uint8(img))
        filename_resized = f"sv_{i}_resized.jpg"
        filename_original = f"sv_{i}_original.jpg"
        pillow_image_resized.save(os.path.join(output_dir, filename_resized))
        pillow_image_original.save(os.path.join(output_dir, filename_original))
    i += 1
for img_name in image_4_sv:
    img = cv2.imread(os.path.join(class_4_path, img_name))
    if img is not None:
        resized_img = cv2.resize(img, (16, 16))
        pillow_image_resized = Image.fromarray(np.uint8(resized_img))
        pillow_image_original = Image.fromarray(np.uint8(img))
        filename_resized = f"sv_{i}_resized.jpg"
        filename_original = f"sv_{i}_original.jpg"
        pillow_image_resized.save(os.path.join(output_dir, filename_resized))
        pillow_image_original.save(os.path.join(output_dir, filename_original))
    i += 1

# %%
import sklearn.svm as svm
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, Y)

# %%
clf.n_support_

# %%
support_vector_indices = clf.support_
support_vectors = clf.support_vectors_
my_indices = []
for i in range(len(alphas)):
    if(alphas[i] > 1e-5):
        my_indices.append(i)

# %%
my_indices_set = set(my_indices)
support_vector_indices_set = set(support_vector_indices)

# Find the common elements (matching indices) using set intersection
matching_indices = my_indices_set.intersection(support_vector_indices_set)

# Get the count of matching indices
num_matching_indices = len(matching_indices)

# %%
num_matching_indices


