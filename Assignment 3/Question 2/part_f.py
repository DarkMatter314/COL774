# %%
# MLP Classifier from scikit learn
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder

# %%
def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y

# %%
def get_metric(y_true, y_pred):
    '''
    Args:
        y_true: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
        y_pred: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
                
    '''
    results = classification_report(y_pred, y_true)
    return results

# %%
x_train_path = './x_train.npy'
y_train_path = './y_train.npy'

X_train, y_train = get_data(x_train_path, y_train_path)

x_test_path = './x_test.npy'
y_test_path = './y_test.npy'

X_test, y_test = get_data(x_test_path, y_test_path)

#you might need one hot encoded y in part a,b,c,d,e
label_encoder = OneHotEncoder(sparse_output = False)
label_encoder.fit(np.expand_dims(y_train, axis = -1))

y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))

# %%
architecures = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
classifiers = []
opfile = open('part_f.txt', 'w')

# %%
for arch in architecures:
    clf = MLPClassifier(solver='sgd', 
                        alpha=0, 
                        hidden_layer_sizes=arch, 
                        activation='relu', 
                        batch_size=32, 
                        learning_rate='invscaling', 
                        learning_rate_init=0.01,
                        verbose=True)
    print(f"Starting architecture: {arch}")
    clf.fit(X_train, y_train)
    print(f"Trained architecture: {arch}")
    classifiers.append(clf)
    opfile.write(f"Architecture: {arch}\n")
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    results_train = get_metric(y_train, y_train_pred)
    results_test = get_metric(y_test, y_test_pred)
    opfile.write(f"Training Data:\n{results_train}\nTesting Data:\n{results_test}\n\n")

# %%
opfile.close()


