# %%
import numpy as np
import pandas as pd

# %%
inputData = pd.read_csv('./Corona_train.csv').to_numpy()

# %%
inputData[0]

# %%
# Features for Naive-Bayes
allWords = []
dictWord = {}
for data in inputData:
    text = data[2].split()
    for word in text:
        if (word != ' ') and (word not in dictWord):
            dictWord[word] = len(allWords)
            allWords.append(word)

# %%
len(dictWord)

# %%
maxValue = 0
i = 0
for data in inputData:
    text = data[2].split(' ')
    dictValues = {}
    for word in text:
        if(word not in dictValues): dictValues[word] = 1
        else: dictValues[word] += 1
    i += 1
    maxValue = max(maxValue, max(dictValues[word] for word in dictValues))

# %%
maxValue

# %%
pc = [0, 0, 0]
pc[0] = inputData[inputData[:, 1] == 'Positive'].shape[0] / inputData.shape[0]
pc[1] = inputData[inputData[:, 1] == 'Neutral'].shape[0] / inputData.shape[0]
pc[2] = inputData[inputData[:, 1] == 'Negative'].shape[0] / inputData.shape[0]

# %%
pc

# %%
def parameters():
    p_wc = np.zeros((3, len(dictWord)))
    for data in inputData:
        text = data[2].split(' ')
        for word in text:
            if word not in dictWord: continue
            if(data[1] == 'Positive'): p_wc[0][dictWord[word]] += 1
            elif(data[1] == 'Neutral'): p_wc[1][dictWord[word]] += 1
            else: p_wc[2][dictWord[word]] += 1
    total = [0, 0, 0]
    total[0] = sum(p_wc[0])
    total[1] = sum(p_wc[1])
    total[2] = sum(p_wc[2])
    for i in range(len(total)):
        for j in range(len(p_wc[i])):
            p_wc[i][j] = (p_wc[i][j] + 1) / (total[i] + len(dictWord))
    return p_wc

# %%
p_wc = parameters()

# %%
def predict(text):
    text = text.split(' ')
    p = [0, 0, 0]
    for i in range(len(p)):
        p[i] = np.log(pc[i])
        for word in text:
            if(word in dictWord):
                p[i] += np.log(p_wc[i][dictWord[word]])
    return np.argmax(p)

# %%
correct = 0
incorrect = 0
for data in inputData:
    print(data[1], end=' ')
    print(predict(data[2]))
    if(data[1] == 'Positive' and predict(data[2]) == 0): correct += 1
    elif(data[1] == 'Neutral' and predict(data[2]) == 1): correct += 1
    elif(data[1] == 'Negative' and predict(data[2]) == 2): correct += 1
    else: incorrect += 1

# %%
print(f"Correct: {correct} \n Incorrecct: {incorrect} \n Accuracy: {correct / (correct + incorrect)}")

# %%
testData = pd.read_csv('./Corona_validation.csv').to_numpy()

# %%
valid_correct = 0
valid_incorrect = 0
for data in testData:
    print(data[1], end=' ')
    print(predict(data[2]))
    if(data[1] == 'Positive' and predict(data[2]) == 0): valid_correct += 1
    elif(data[1] == 'Neutral' and predict(data[2]) == 1): valid_correct += 1
    elif(data[1] == 'Negative' and predict(data[2]) == 2): valid_correct += 1
    else: valid_incorrect += 1

# %%
print(f"Correct: {valid_correct} \n Incorrect: {valid_incorrect} \n Accuracy: {valid_correct / (valid_correct + valid_incorrect)}")

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = ''.join(inputData[:, 2])
text_positive = ''.join(inputData[inputData[:, 1] == 'Positive'][:, 2])
text_neutral = ''.join(inputData[inputData[:, 1] == 'Neutral'][:, 2])
text_negative = ''.join(inputData[inputData[:, 1] == 'Negative'][:, 2])
word_cloud = WordCloud(width = 800, height = 500, background_color ='white', min_font_size = 10).generate(text)
word_cloud_positive = WordCloud(width = 800, height = 500, background_color ='white', min_font_size = 10).generate(text_positive)
word_cloud_neutral = WordCloud(width = 800, height = 500, background_color ='white', min_font_size = 10).generate(text_neutral)
word_cloud_negative = WordCloud(width = 800, height = 500, background_color ='white', min_font_size = 10).generate(text_negative)

# %%
plt.figure(figsize=(10, 5))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
word_cloud.to_file("wordcloud_a_all.png")
word_cloud_positive.to_file("wordcloud_a_positive.png")
word_cloud_neutral.to_file("wordcloud_a_neutral.png")
word_cloud_negative.to_file("wordcloud_a_negative.png")

# %%
import random

def random_predict(text):
    return random.randint(0, 2)

def always_positive_predict(text):
    return 0

random_correct = 0
random_incorrect = 0
for data in testData:
    # print(data[1], end=' ')
    # print(random_predict(data[2]))
    if(data[1] == 'Positive' and random_predict(data[2]) == 0): random_correct += 1
    elif(data[1] == 'Neutral' and random_predict(data[2]) == 1): random_correct += 1
    elif(data[1] == 'Negative' and random_predict(data[2]) == 2): random_correct += 1
    else: random_incorrect += 1

print("Random Prediction:")
print(f"Correct: {random_correct} \n Incorrect: {random_incorrect} \n Accuracy: {random_correct / (random_correct + random_incorrect)}")

always_positive_correct = 0
always_positive_incorrect = 0
for data in testData:
    # print(data[1], end=' ')
    # print(always_positive_predict(data[2]))
    if(data[1] == 'Positive' and always_positive_predict(data[2]) == 0): always_positive_correct += 1
    elif(data[1] == 'Neutral' and always_positive_predict(data[2]) == 1): always_positive_correct += 1
    elif(data[1] == 'Negative' and always_positive_predict(data[2]) == 2): always_positive_correct += 1
    else: always_positive_incorrect += 1

print("Always Positive Prediction:")
print(f"Correct: {always_positive_correct} \n Incorrect: {always_positive_incorrect} \n Accuracy: {always_positive_correct / (always_positive_correct + always_positive_incorrect)}")

# %%
# Confusion Matrix

print("Confusion Matrix for Naive Bayes Model:")
confusion_matrix_model_train = np.zeros((3, 3))
for data in inputData:
    prediction = predict(data[2])
    if(data[1] == 'Positive'): confusion_matrix_model_train[0][prediction] += 1
    elif(data[1] == 'Neutral'): confusion_matrix_model_train[1][prediction] += 1
    else: confusion_matrix_model_train[2][prediction] += 1

print("1. Training Data")
print(confusion_matrix_model_train)

confusion_matrix_model_valid = np.zeros((3, 3))
for data in testData:
    prediction = predict(data[2])
    if(data[1] == 'Positive'): confusion_matrix_model_valid[0][prediction] += 1
    elif(data[1] == 'Neutral'): confusion_matrix_model_valid[1][prediction] += 1
    else: confusion_matrix_model_valid[2][prediction] += 1

print("2. Validation Data")
print(confusion_matrix_model_valid)

print("Confusion Matrix for Random Prediction Model:")
confusion_matrix_random_train = np.zeros((3, 3))
for data in inputData:
    prediction = random_predict(data[2])
    if(data[1] == 'Positive'): confusion_matrix_random_train[0][prediction] += 1
    elif(data[1] == 'Neutral'): confusion_matrix_random_train[1][prediction] += 1
    else: confusion_matrix_random_train[2][prediction] += 1

print("1. Training Data")
print(confusion_matrix_random_train)

confusion_matrix_random_valid = np.zeros((3, 3))
for data in testData:
    prediction = random_predict(data[2])
    if(data[1] == 'Positive'): confusion_matrix_random_valid[0][prediction] += 1
    elif(data[1] == 'Neutral'): confusion_matrix_random_valid[1][prediction] += 1
    else: confusion_matrix_random_valid[2][prediction] += 1

print("2. Validation Data")
print(confusion_matrix_random_valid)

print("Confusion Matrix for Always Positive Prediction Model:")
confusion_matrix_always_positive_train = np.zeros((3, 3))
for data in inputData:
    prediction = always_positive_predict(data[2])
    if(data[1] == 'Positive'): confusion_matrix_always_positive_train[0][prediction] += 1
    elif(data[1] == 'Neutral'): confusion_matrix_always_positive_train[1][prediction] += 1
    else: confusion_matrix_always_positive_train[2][prediction] += 1

print("1. Training Data")
print(confusion_matrix_always_positive_train)

confusion_matrix_always_positive_valid = np.zeros((3, 3))
for data in testData:
    prediction = always_positive_predict(data[2])
    if(data[1] == 'Positive'): confusion_matrix_always_positive_valid[0][prediction] += 1
    elif(data[1] == 'Neutral'): confusion_matrix_always_positive_valid[1][prediction] += 1
    else: confusion_matrix_always_positive_valid[2][prediction] += 1

print("2. Validation Data")
print(confusion_matrix_always_positive_valid)



