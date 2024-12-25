# %%
import numpy as np
import pandas as pd
import os
import cv2
import sklearn.svm as svm

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
X = np.array(image_3_list + image_4_list)
Y = np.array([-1] * len(image_3_list) + [1] * len(image_4_list))

# %%
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, Y)


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
    prediction = clf.predict([img])
    if(prediction[0] == -1):
        correct += 1
    else:
        incorrect += 1
for img in validation_image_4_list:
    prediction = clf.predict([img])
    if(prediction[0] == 1):
        correct += 1
    else:
        incorrect += 1
print(f"Correct: {correct} \nIncorrect: {incorrect} \nAccuracy: {correct / (correct + incorrect)}")

# %%
clf.n_support_

# %%
clf_gaussian = svm.SVC(kernel='rbf', C=1.0)
clf_gaussian.fit(X, Y)

# %%
rbf_correct = 0
rbf_incorrect = 0
for img in validation_image_3_list:
    prediction = clf_gaussian.predict([img])
    if(prediction[0] == -1):
        rbf_correct += 1
    else:
        rbf_incorrect += 1
for img in validation_image_4_list:
    prediction = clf_gaussian.predict([img])
    if(prediction[0] == 1):
        rbf_correct += 1
    else:
        rbf_incorrect += 1
print(f"Correct: {rbf_correct} \nIncorrect: {rbf_incorrect} \nAccuracy: {rbf_correct / (rbf_correct + rbf_incorrect)}")

# %%
clf_gaussian.n_support_


