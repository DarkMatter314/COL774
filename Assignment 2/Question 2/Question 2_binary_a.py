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
XY = Y.reshape(-1, 1) * X
H = np.dot(XY, XY.T)

# %%
P = matrix(H, tc='d')
q = matrix(-np.ones((m, 1)), tc='d')
G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))), tc='d')
h = matrix(np.hstack((np.zeros(m), np.ones(m) * 1)), tc='d')
A = matrix(Y.reshape(1, -1), tc='d')
b = matrix(np.zeros(1), tc='d')

# %%
Y.reshape(1, -1).shape

# %%
sol = solvers.qp(P, q, G, h, A, b)

# %%
solution = np.array(sol['x'])

# %%
w = np.sum(solution * Y.reshape(-1, 1) * X, axis=0)
b = np.sum(solution * Y.reshape(-1, 1) * X @ X[0, :].reshape(-1, 1)) - Y[0]

# %%
def predict(x, w, b):
    val = w.T @ x + b
    return 1 if val >= 0 else -1

# %%
incorrect = 0
for img in image_3_list:
    prediction = predict(img, w, b)
    if(prediction != -1):
        incorrect += 1
for img in image_4_list:
    prediction = predict(img, w, b)
    if(prediction != 1):
        incorrect += 1
print(f"Correct: {m - incorrect}\nIncorrect: {incorrect}\nAccuracy: {(m - incorrect) / m}")

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
for img in validation_image_3_list:
    prediction = predict(img, w, b)
    if(prediction == -1):
        correct += 1
    else:
        incorrect += 1
for img in validation_image_4_list:
    prediction = predict(img, w, b)
    if(prediction == 1):
        correct += 1
    else:
        incorrect += 1
print(f"Correct: {correct} \nIncorrect: {incorrect} \nAccuracy: {correct / (correct + incorrect)}")

# %%
num_sv = len(solution[solution > 1e-5])
num_sv

# %%
num_sv/len(solution)

# %%
sorted_indices = np.argsort(solution, axis=0)[::-1]
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
top_6_support_vectors

# %%
from PIL import Image
output_dir = './part_a_sv/'
i = 0
for img_name in image_3_sv:
    img = cv2.imread(os.path.join(class_3_path, img_name))
    if img is not None:
        resized_img = cv2.resize(img, (16, 16))
        pillow_image_resized = Image.fromarray(np.uint8(resized_img))
        pillow_image_original = Image.fromarray(np.uint8(img))
        filename_resized = f"image_{i}_sv_resized.jpg"
        filename_original = f"image_{i}_sv_original.jpg"
        pillow_image_resized.save(os.path.join(output_dir, filename_resized))
        pillow_image_original.save(os.path.join(output_dir, filename_original))
    i += 1
for img_name in image_4_sv:
    img = cv2.imread(os.path.join(class_4_path, img_name))
    if img is not None:
        resized_img = cv2.resize(img, (16, 16))
        pillow_image_resized = Image.fromarray(np.uint8(resized_img))
        pillow_image_original = Image.fromarray(np.uint8(img))
        filename_resized = f"image_{i}_sv_resized.jpg"
        filename_original = f"image_{i}_sv_original.jpg"
        pillow_image_resized.save(os.path.join(output_dir, filename_resized))
        pillow_image_original.save(os.path.join(output_dir, filename_original))
    i += 1

# %%
import matplotlib.pyplot as plt
w_img = w.reshape(16, 16, 3)
min_w = np.min(w_img)
max_w = np.max(w_img)
cmap = 'viridis'
normalized_w = (w_img - min_w) / (max_w - min_w)
fig, ax = plt.subplots()
im = ax.imshow(normalized_w, cmap=cmap)
ax.axis('off')
fig.canvas.draw()
pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(),
                            fig.canvas.tostring_rgb())
pil_image.save('output_image.jpg')

# %%
import sklearn.svm as svm
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, Y)

# %%
np.linalg.norm(clf.coef_ - w)

# %%
clf.intercept_ - b

# %%
support_vector_indices = clf.support_
support_vectors = clf.support_vectors_
my_indices = []
for i in range(len(solution)):
    if(solution[i] > 1e-5):
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


