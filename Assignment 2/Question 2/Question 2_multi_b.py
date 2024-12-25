# %%
import numpy as np
import pandas as pd
import os
import cv2
import sklearn.svm as svm

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
X = np.array(images[0] + images[1] + images[2] + images[3] + images[4] + images[5])
Y = np.array([0] * len(images[0]) + [1] * len(images[1]) + [2] * len(images[2]) + [3]*len(images[3]) + [4]*len(images[4]) + [5]*len(images[5]))

# %%
svm_skl = svm.SVC(kernel='rbf', C=1.0, decision_function_shape='ovo')
svm_skl.fit(X, Y)

# %%
svm_valid_correct = 0
svm_valid_incorrect = 0
misclassified = []
misclassified_class = [[] for i in range(6)]
misclassified_prediction = [[] for i in range(6)]
svm_confusion_matrix = np.zeros((6, 6))
for i in range(len(validation_images)):
    print(f"Starting validation for class {i}")
    j = 0
    for img in validation_images[i]:
        print(j, end=' ')
        prediction = svm_skl.predict([img])[0]
        svm_confusion_matrix[i][prediction] += 1
        if prediction == i:
            svm_valid_correct += 1
        else:
            svm_valid_incorrect += 1
            misclassified.append(img)
            misclassified_class[i].append(j)
            misclassified_prediction[i].append(prediction)
        j += 1
    print()

# %%
print(f"Correct: {svm_valid_correct}\nIncorrect: {svm_valid_incorrect}\nAccuracy: {svm_valid_correct / (svm_valid_correct + svm_valid_incorrect)}")
print(f"Confusion Matrix:\n{svm_confusion_matrix}")


