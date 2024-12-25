# %%
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from string import punctuation

# %%
training_data = pd.read_csv('./Corona_train.csv').to_numpy()
transformed_data = pd.read_csv('./Corona_train.csv').to_numpy()
validation_data = pd.read_csv('./Corona_validation.csv').to_numpy()

# %%
stemmer = PorterStemmer()
stopwords_set = stopwords.words('english')

# %%
def stem_punc_stopwords(data):
    for i in range(len(data)):
        text = data[i][2].lower()
        words = word_tokenize(text)
        words = [word for word in words if word not in punctuation]
        changed_words = [stemmer.stem(word) for word in words if word not in stopwords_set]
        data[i][2] = ' '.join(changed_words)
    return data

# %%
transformed_data = stem_punc_stopwords(transformed_data)

# %%
def get_word_frequency(inputData):
    allWords = []
    dictWord = {}
    for data in inputData:
        text = word_tokenize(data[2])
        for word in text:
            if (word != ' ') and (word not in dictWord):
                dictWord[word] = len(allWords)
                allWords.append(word)
    return (allWords, dictWord)

# %%
def get_bigrams(inputData):
    bigrams_list = []
    bigrams_dict = {}
    for data in inputData:
        text = word_tokenize(data[2])
        for i in range(len(text) - 1):
            bigram = text[i] + ' ' + text[i + 1]
            if (bigram != ' ') and (bigram not in bigrams_dict):
                bigrams_dict[bigram] = len(bigrams_list)
                bigrams_list.append(bigram)
    return (bigrams_list, bigrams_dict)

# %%
(all_words, dict_word) = get_word_frequency(transformed_data)
(bigrams_list, dict_bigrams) = get_bigrams(transformed_data)

# %%
pc = np.zeros((3,))
pc[0] = training_data[training_data[:, 1] == 'Positive'].shape[0] / training_data.shape[0]
pc[1] = training_data[training_data[:, 1] == 'Neutral'].shape[0] / training_data.shape[0]
pc[2] = training_data[training_data[:, 1] == 'Negative'].shape[0] / training_data.shape[0]

# %%
def parameters(inputData, dictWord, dict_bigrams):
    p_wc = np.zeros((3, len(dictWord)))
    for data in inputData:
        text = word_tokenize(data[2])
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
    p_wc_bigrams = np.zeros((3, len(dict_bigrams)))
    for data in inputData:
        text = word_tokenize(data[2])
        for i in range(len(text) - 1):
            bigram = text[i] + ' ' + text[i + 1]
            if bigram not in dict_bigrams: continue
            if(data[1] == 'Positive'): p_wc_bigrams[0][dict_bigrams[bigram]] += 1
            elif(data[1] == 'Neutral'): p_wc_bigrams[1][dict_bigrams[bigram]] += 1
            else: p_wc_bigrams[2][dict_bigrams[bigram]] += 1
    total_bigrams = [0, 0, 0]
    total_bigrams[0] = sum(p_wc_bigrams[0])
    total_bigrams[1] = sum(p_wc_bigrams[1])
    total_bigrams[2] = sum(p_wc_bigrams[2])
    for i in range(len(total_bigrams)):
        for j in range(len(p_wc_bigrams[i])):
            p_wc_bigrams[i][j] = (p_wc_bigrams[i][j] + 1) / (total_bigrams[i] + len(dict_bigrams))
    return (p_wc, p_wc_bigrams)

# %%
(p_wc_single, p_wc_bigrams) = parameters(transformed_data, dict_word, dict_bigrams)

# %%
def bigram_predict(text, dict_single, dict_bigrams, p_wc_single, p_wc_bigrams):
    text = text.lower()
    text = word_tokenize(text)
    words = [stemmer.stem(word) for word in text if ((word not in punctuation) and (word not in stopwords_set))]
    p = np.zeros((3,))
    for i in range(len(p)):
        p[i] = np.log(pc[i])
        for word in words:
            if(word in dict_single):
                p[i] += np.log(p_wc_single[i][dict_single[word]])
        for j in range(len(words) - 1):
            bigram = words[j] + ' ' + words[j + 1]
            if(bigram in dict_bigrams):
                p[i] += np.log(p_wc_bigrams[i][dict_bigrams[bigram]])
    return np.argmax(p)

# %%
train_correct = 0
train_incorrect = 0
for data in training_data:
    prediction = bigram_predict(data[2], dict_word, dict_bigrams, p_wc_single, p_wc_bigrams)
    if(prediction == 0 and data[1] == 'Positive'): train_correct += 1
    elif(prediction == 1 and data[1] == 'Neutral'): train_correct += 1
    elif(prediction == 2 and data[1] == 'Negative'): train_correct += 1
    else: train_incorrect += 1

valid_correct = 0
valid_incorrect = 0
for data in validation_data:
    prediction = bigram_predict(data[2], dict_word, dict_bigrams, p_wc_single, p_wc_bigrams)
    if(prediction == 0 and data[1] == 'Positive'): valid_correct += 1
    elif(prediction == 1 and data[1] == 'Neutral'): valid_correct += 1
    elif(prediction == 2 and data[1] == 'Negative'): valid_correct += 1
    else: valid_incorrect += 1

# %%
print(f"Training\nCorrect: {train_correct}\nIncorrect: {train_incorrect}\nAccuracy: {train_correct / (train_correct + train_incorrect)}")
print(f"Validation\nCorrect: {valid_correct}\nIncorrect: {valid_incorrect}\nAccuracy: {valid_correct / (valid_correct + valid_incorrect)}")


