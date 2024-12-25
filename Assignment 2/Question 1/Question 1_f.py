# %%
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation

# %%
training_data = pd.read_csv('./Corona_train.csv').to_numpy()
domain_1_data = pd.read_csv('./Domain_Adaptation/Twitter_train_1.csv').to_numpy()
domain_2_data = pd.read_csv('./Domain_Adaptation/Twitter_train_2.csv').to_numpy()
domain_5_data = pd.read_csv('./Domain_Adaptation/Twitter_train_5.csv').to_numpy()
domain_10_data = pd.read_csv('./Domain_Adaptation/Twitter_train_10.csv').to_numpy()
domain_25_data = pd.read_csv('./Domain_Adaptation/Twitter_train_25.csv').to_numpy()
domain_50_data = pd.read_csv('./Domain_Adaptation/Twitter_train_50.csv').to_numpy()
domain_100_data = pd.read_csv('./Domain_Adaptation/Twitter_train_100.csv').to_numpy()
domain_validation_data = pd.read_csv('./Domain_Adaptation/Twitter_validation.csv').to_numpy()
stemmer = PorterStemmer()
stopwords_set = stopwords.words('english')

# %%
data = [domain_1_data, domain_2_data, domain_5_data, domain_10_data, domain_25_data, domain_50_data, domain_100_data]
data_name = ['domain_1', 'domain_2', 'domain_5', 'domain_10', 'domain_25', 'domain_50', 'domain_100']

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
def parameters(inputData, dictWord):
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
    return p_wc

# %%
def stem_punc_predict(text, dictWord, pc, p_wc):
    text = text.lower()
    text = word_tokenize(text)
    words = [word for word in text if word not in punctuation]
    p = np.zeros((3,))
    for i in range(len(p)):
        p[i] = np.log(pc[i])
        for word in words:
            if(word in stopwords_set): continue
            word = stemmer.stem(word)
            if(word in dictWord):
                p[i] += np.log(p_wc[i][dictWord[word]])
    return np.argmax(p)

# %%
file = open('q1_only_domain.txt', 'w')

# %%
data[1].shape

# %%
for i in range(7):
    print(i)
    X = []
    if(i==0): X = domain_1_data
    if(i==1): X = domain_2_data
    if(i==2): X = domain_5_data
    if(i==3): X = domain_10_data
    if(i==4): X = domain_25_data
    if(i==5): X = domain_50_data
    if(i==6): X = domain_100_data
    X = stem_punc_stopwords(X)
    (allWords, dictWord) = get_word_frequency(X)
    file.write(f"Domain: {data_name[i]}\n")
    pc = [0, 0, 0]
    pc[0] = X[X[:, 1] == 'Positive'].shape[0] / X.shape[0]
    pc[1] = X[X[:, 1] == 'Neutral'].shape[0] / X.shape[0]
    pc[2] = X[X[:, 1] == 'Negative'].shape[0] / X.shape[0]
    p_wc = parameters(X, dictWord)
    correct = 0
    incorrect = 0
    for data in domain_validation_data:
        prediction = stem_punc_predict(data[2], dictWord, pc, p_wc)
        if(data[1] == 'Positive' and prediction == 0): correct += 1
        elif(data[1] == 'Neutral' and prediction == 1): correct += 1
        elif(data[1] == 'Negative' and prediction == 2): correct += 1
        else: incorrect += 1
    file.write(f"Correct: {correct}\nIncorrect: {incorrect}\nAccuracy: {correct / (correct + incorrect)}\n")
file.close()


