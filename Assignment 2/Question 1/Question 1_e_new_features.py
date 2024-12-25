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
validation_data = pd.read_csv('./Corona_validation.csv').to_numpy()

# %%
stemmer = PorterStemmer()
stopwords_set = stopwords.words('english')

# %%
pc = np.zeros((3,))
pc[0] = training_data[training_data[:, 1] == 'Positive'].shape[0] / training_data.shape[0]
pc[1] = training_data[training_data[:, 1] == 'Neutral'].shape[0] / training_data.shape[0]
pc[2] = training_data[training_data[:, 1] == 'Negative'].shape[0] / training_data.shape[0]

# %%
p_extra = np.zeros((3, 4))
train_positive = training_data[training_data[:, 1] == 'Positive']
train_neutral = training_data[training_data[:, 1] == 'Neutral']
train_negative = training_data[training_data[:, 1] == 'Negative']
# 1 - Exclamation Marks
positive_exclam = np.sum(np.array([data[2].count('!') for data in train_positive]))
neutral_exclam = np.sum(np.array([data[2].count('!') for data in train_neutral]))
negative_exclam = np.sum(np.array([data[2].count('!') for data in train_negative]))
total_exclam = positive_exclam  + neutral_exclam + negative_exclam
p_extra[0][1] = positive_exclam / total_exclam
p_extra[1][1] = neutral_exclam / total_exclam
p_extra[2][1] = negative_exclam / total_exclam

# 2 - Hashtags
positive_hashtags = np.sum(np.array([data[2].count('#') for data in train_positive]))
neutral_hashtags = np.sum(np.array([data[2].count('#') for data in train_neutral]))
negative_hashtags = np.sum(np.array([data[2].count('#') for data in train_negative]))
total_hashtags = positive_hashtags + neutral_hashtags + negative_hashtags
p_extra[0][2] = positive_hashtags / total_hashtags
p_extra[1][2] = neutral_hashtags / total_hashtags
p_extra[2][2] = negative_hashtags / total_hashtags

# 3 - Question Marks
positive_quest = np.sum(np.array([data[2].count('?') for data in train_positive]))
neutral_quest = np.sum(np.array([data[2].count('?') for data in train_neutral]))
negative_quest = np.sum(np.array([data[2].count('?') for data in train_negative]))
total_quest = positive_quest + neutral_quest + negative_quest
p_extra[0][3] = positive_quest / total_quest
p_extra[1][3] = neutral_quest / total_quest
p_extra[2][3] = negative_quest / total_quest


# 3 - Uppercase Letter
# positive_uppercase_frequency = np.sum(np.array([]))

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
new_training_data = stem_punc_stopwords(training_data)
new_train_positive = new_training_data[new_training_data[:, 1] == 'Positive']
new_train_neutral = new_training_data[new_training_data[:, 1] == 'Neutral']
new_train_negative = new_training_data[new_training_data[:, 1] == 'Negative']
text_length_positive = np.sum(np.array([len(data[2]) for data in new_train_positive]))
text_length_neutral = np.sum(np.array([len(data[2]) for data in new_train_neutral]))
text_length_negative = np.sum(np.array([len(data[2]) for data in new_train_negative]))
total_text_length = text_length_positive + text_length_negative + text_length_neutral
p_extra[0][0] = text_length_positive / total_text_length
p_extra[1][0] = text_length_neutral / total_text_length
p_extra[2][0] = text_length_negative / total_text_length

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
(stem_punc_words, stem_punc_dict) = get_word_frequency(new_training_data)

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
p_wc_stem_punc = parameters(new_training_data, stem_punc_dict)

# %%
def stem_punc_predict(text, dictWord, pc, p_wc, p_extra):
    p = np.zeros((3,))
    # 0 - Length, 1 - !, 2 - #, 3 - ?
    text_exclam = text.count('!')
    text_quest = text.count('?')
    text_hash = text.count('#')
    lowered_text = text.lower()
    lowered_text = word_tokenize(lowered_text)
    upper_text = word_tokenize(text)
    uppercase_words = []
    words = []
    for i in range(len(lowered_text)):
        if(lowered_text[i] in punctuation): continue
        uppercase_words.append(upper_text[i])
        words.append(lowered_text[i])
    for i in range(len(p)):
        p[i] = np.log(pc[i])
        p[i] += text_exclam * p_extra[i][1]
        p[i] += text_quest * p_extra[i][3]
        p[i] += text_hash * p_extra[i][2]
        # p[i] += text_len * p_extra[i][0]
        for j in range(len(words)):
            if(words[j] in stopwords_set): continue
            capitalised = 0
            for char in uppercase_words[j]:
                if(char >= 'A' and char <= 'Z'): capitalised += 1
            total = len(uppercase_words[j])
            multiplier = (1 + capitalised/total)
            words[j] = stemmer.stem(words[j])
            if(words[j] in dictWord):
                p[i] += multiplier * np.log(p_wc[i][dictWord[words[j]]])
    return np.argmax(p)

# %%
stem_punc_train_correct = 0
stem_punc_train_incorrect = 0
for data in training_data:
    prediction = stem_punc_predict(data[2], stem_punc_dict, pc, p_wc_stem_punc, p_extra)
    if(data[1] == 'Positive' and prediction == 0): stem_punc_train_correct += 1
    elif(data[1] == 'Neutral' and prediction == 1): stem_punc_train_correct += 1
    elif(data[1] == 'Negative' and prediction == 2): stem_punc_train_correct += 1
    else: stem_punc_train_incorrect += 1
stem_punc_val_correct = 0
stem_punc_val_incorrect = 0
for data in validation_data:
    prediction = stem_punc_predict(data[2], stem_punc_dict, pc, p_wc_stem_punc, p_extra)
    if(data[1] == 'Positive' and prediction == 0): stem_punc_val_correct += 1
    elif(data[1] == 'Neutral' and prediction == 1): stem_punc_val_correct += 1
    elif(data[1] == 'Negative' and prediction == 2): stem_punc_val_correct += 1
    else: stem_punc_val_incorrect += 1

# %%
print(f"New Features, Training\nCorrect: {stem_punc_train_correct}\nIncorrect: {stem_punc_train_incorrect}\nAccuracy: {stem_punc_train_correct / (stem_punc_train_correct + stem_punc_train_incorrect)}")
print(f"New Features, Validation\nCorrect: {stem_punc_val_correct}\nIncorrect: {stem_punc_val_incorrect}\nAccuracy: {stem_punc_val_correct / (stem_punc_val_correct + stem_punc_val_incorrect)}")

# %%
p_extra


