# %%
import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
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
    print(results)

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
y_test_new = np.argmax(y_test_onehot, axis = 1)
y_train_new = np.argmax(y_train_onehot, axis = 1)

# %%
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# %%
def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

# %%
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)

# %%
def relu(x):
    return np.maximum(0, x)

# %%
def relu_derivative(x):
    # if x = 0 derivative is 0.5
    y = np.copy(x)
    y[x < 0] = 0
    y[x == 0] = 0.5
    y[x > 0] = 1
    return y

# %%
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        self.activation = activation
        self.dW = np.zeros((input_size, output_size))
        self.db = np.zeros(output_size)
        self.da = np.zeros(input_size)
        self.dz = np.zeros(output_size)
        self.x = np.zeros(input_size)
        self.z = np.zeros(output_size)
        self.a = np.zeros(input_size)

    def linear(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def forward(self, x):
        self.z = self.linear(x)
        if(self.activation == 0): self.a = sigmoid(self.z)
        else: self.a = relu(self.z)
        return self.a
    
    def backward(self, da):
        if(self.activation == 0): self.dz = da * sigmoid_derivative(self.z)
        else: self.dz = da * relu_derivative(self.z)
        self.dW = np.dot(self.x.T, self.dz)
        self.db = np.sum(self.dz, axis = 0)
        self.da = np.dot(self.dz, self.W.T)
        return self.da 

    def opbackward(self, dz):
        self.dz = dz
        self.dW = np.dot(self.x.T, self.dz)
        self.db = np.sum(self.dz, axis = 0)
        self.da = np.dot(self.dz, self.W.T)
        return self.da 
    
    def update(self, lr, batch_size):
        self.W -= lr * self.dW / batch_size
        self.b -= lr * self.db / batch_size

    def clearGradients(self):
        self.dW = np.zeros((self.input_size, self.output_size))
        self.db = np.zeros(self.output_size)

# %%
class NeuralNetwork:

    def __init__(self, input_size, output_size, hidden_sizes, activation = 0, adaptive_lr = False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.layers = []
        self.losses = []
        self.accuracies = []
        self.activation = activation
        self.adaptive_lr = adaptive_lr
        if(len(hidden_sizes) == 0):
            self.layers.append(Layer(input_size, output_size, activation))
            return
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(Layer(input_size, hidden_sizes[i], activation))
            else:
                self.layers.append(Layer(hidden_sizes[i-1], hidden_sizes[i], activation))
        self.layers.append(Layer(hidden_sizes[-1], output_size, activation))

    def predict(self, x):
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return softmax(self.layers[-1].linear(x))
    
    def loss(self, y_pred, y_true):
        # cross entropy loss
        return - np.sum(y_true * np.log(y_pred + 1e-8)) / len(y_pred)
    
    def evaluate_set(self, x, y):
        y_pred = self.predict(x)
        y_pred = np.argmax(y_pred, axis = 1)
        y_true = np.argmax(y, axis = 1)
        return np.sum(y_pred == y_true) / len(y_true)
    
    def backpropogate(self, x, y, lr = 0.01, batch_size = 32):
        y_pred = self.predict(x)
        dz = y_pred - y
        da = self.layers[-1].opbackward(dz)
        for layer in reversed(self.layers[:-1]):
            da = layer.backward(da)
        for layer in self.layers:
            layer.update(lr, batch_size)
        for layer in self.layers:
            layer.clearGradients()

    def train(self, x_train, y_train, epochs = 1000, mini_batch_size = 32, lr = 0.01):
        prev_loss = 0
        new_loss = 0
        for epoch in range(epochs):
            for i in range(0, len(x_train), mini_batch_size):
                x_batch = x_train[i:i+mini_batch_size]
                y_batch = y_train[i:i+mini_batch_size]
                y_pred = self.predict(x_batch)
                prev_loss = new_loss
                new_loss = self.loss(y_pred, y_batch)
                # if(prev_loss - new_loss < 1e-6 and epoch > 100):
                #     break
                if(self.adaptive_lr == True): self.backpropogate(x_batch, y_batch, lr/sqrt(epoch+1))
                else: self.backpropogate(x_batch, y_batch, lr)
            if epoch % 10 == 0:
                # train_accuracy = self.evaluate_set(x_train, y_train)
                # test_accuracy = self.evaluate_set(X_test, y_test_onehot)
                # self.accuracies.append((epoch, train_accuracy, test_accuracy))
                print(f"Epoch = {epoch}")
            # if(prev_loss - new_loss < 1e-6 and epoch > 100):
            #     break
            # if epoch % 10 == 0:
                # print("Epoch: {}, Loss: {}".format(epoch, loss))
                # self.accuracies.append(self.evaluate(x, y))

# %%
node_sizes = [1, 5, 10, 100]
outfile = open('part_b_all.txt', 'w')

for node_size in node_sizes:
    nn_node = NeuralNetwork(1024, 5, [node_size])
    print(f"Starting Node Size: {node_size}")
    nn_node.train(X_train, y_train_onehot, epochs = 1000, mini_batch_size = 32, lr = 0.01)
    print(f"Trained Node Size: {node_size}")
    y_pred = nn_node.predict(X_test)
    y_pred_new = np.argmax(y_pred, axis = 1)
    precision = precision_score(y_test_new, y_pred_new, average=None)
    recall = recall_score(y_test_new, y_pred_new, average=None)
    f1 = f1_score(y_test_new, y_pred_new, average=None)
    accuracy = np.sum(y_pred_new == y_test_new) / len(y_test_new)
    print(f"Accuracy: {accuracy}")
    outfile.write(f"Node Size: {node_size}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nAccuracy: {accuracy}\n\n")

# %%
outfile.close()

# %%
part_c_file = open('part_c_last.txt', 'w')
architectures = [[512, 256, 128, 64]]

for architecture in architectures:
    nn_arch = NeuralNetwork(1024, 5, architecture)
    print(f"Starting Architecture: {architecture}")
    nn_arch.train(X_train, y_train_onehot, epochs = 1000, mini_batch_size = 32, lr = 0.01)
    print(f"Trained Architecture: {architecture}")
    y_pred = nn_arch.predict(X_test)
    y_pred_new = np.argmax(y_pred, axis = 1)
    precision = precision_score(y_test_new, y_pred_new, average=None)
    recall = recall_score(y_test_new, y_pred_new, average=None)
    f1 = f1_score(y_test_new, y_pred_new, average=None)
    accuracy = np.sum(y_pred_new == y_test_new) / len(y_test_new)
    print(f"Accuracy: {accuracy}")
    part_c_file.write(f"Architecture: {architecture}\nTest Data\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nAccuracy: {accuracy}\n")
    y_train_pred = nn_arch.predict(X_train)
    y_train_pred_new = np.argmax(y_train_pred, axis = 1)
    precision = precision_score(y_train, y_train_pred_new, average=None)
    recall = recall_score(y_train, y_train_pred_new, average=None)
    f1 = f1_score(y_train, y_train_pred_new, average=None)
    accuracy = accuracy_score(y_train, y_train_pred_new)
    print(f"Train Accuracy: {accuracy}")
    part_c_file.write(f"Training Data\nPrecision: {precision}\nTrain Recall: {recall}\nTrain F1 Score: {f1}\nTrain Accuracy: {accuracy}\n\n")
part_c_file.close()

# %%
part_c_file.close()

# %%
# Adaptive LR and Sigmoid
part_d_file = open('part_d.txt', 'w')
architectures = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]

for architecture in architectures:
    nn_arch = NeuralNetwork(1024, 5, architecture, 0, True)
    print(f"Starting Architecture: {architecture}")
    nn_arch.train(X_train, y_train_onehot, epochs = 1000, mini_batch_size = 32, lr = 0.01)
    print(f"Trained Architecture: {architecture}")
    y_pred = nn_arch.predict(X_test)
    y_pred_new = np.argmax(y_pred, axis = 1)
    precision = precision_score(y_test_new, y_pred_new, average=None)
    recall = recall_score(y_test_new, y_pred_new, average=None)
    f1 = f1_score(y_test_new, y_pred_new, average=None)
    accuracy = np.sum(y_pred_new == y_test_new) / len(y_test_new)
    print(f"Accuracy: {accuracy}")
    part_d_file.write(f"Architecture: {architecture}\nTest Data\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nAccuracy: {accuracy}\n")
    y_train_pred = nn_arch.predict(X_train)
    y_train_pred_new = np.argmax(y_train_pred, axis = 1)
    precision = precision_score(y_train_new, y_train_pred_new, average=None)
    recall = recall_score(y_train_new, y_train_pred_new, average=None)
    f1 = f1_score(y_train_new, y_train_pred_new, average=None)
    accuracy = accuracy_score(y_train_new, y_train_pred_new)
    print(f"Train Accuracy: {accuracy}")
    part_d_file.write(f"Training Data\nPrecision: {precision}\nTrain Recall: {recall}\nTrain F1 Score: {f1}\nTrain Accuracy: {accuracy}\n\n")
part_d_file.close()

# %%
# Adaptive LR and ReLu 
part_e_file = open('part_e.txt', 'w')
architectures = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]

for architecture in architectures:
    nn_arch = NeuralNetwork(1024, 5, architecture, 0, True)
    print(f"Starting Architecture: {architecture}")
    nn_arch.train(X_train, y_train_onehot, epochs = 1000, mini_batch_size = 32, lr = 0.01)
    print(f"Trained Architecture: {architecture}")
    y_pred = nn_arch.predict(X_test)
    y_pred_new = np.argmax(y_pred, axis = 1)
    precision = precision_score(y_test_new, y_pred_new, average=None)
    recall = recall_score(y_test_new, y_pred_new, average=None)
    f1 = f1_score(y_test_new, y_pred_new, average=None)
    accuracy = np.sum(y_pred_new == y_test_new) / len(y_test_new)
    print(f"Accuracy: {accuracy}")
    part_e_file.write(f"Architecture: {architecture}\nTest Data\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nAccuracy: {accuracy}\n")
    y_train_pred = nn_arch.predict(X_train)
    y_train_pred_new = np.argmax(y_train_pred, axis = 1)
    precision = precision_score(y_train_new, y_train_pred_new, average=None)
    recall = recall_score(y_train_new, y_train_pred_new, average=None)
    f1 = f1_score(y_train_new, y_train_pred_new, average=None)
    accuracy = accuracy_score(y_train_new, y_train_pred_new)
    print(f"Train Accuracy: {accuracy}")
    part_e_file.write(f"Training Data\nPrecision: {precision}\nTrain Recall: {recall}\nTrain F1 Score: {f1}\nTrain Accuracy: {accuracy}\n\n")
part_e_file.close()


