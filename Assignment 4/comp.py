import pandas as pd
import numpy as np  
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, csv
device = 'cuda' if torch.cuda.is_available() else 'cpu'

path_data = sys.argv[1]
traincsv = pd.read_csv(f'{path_data}/SyntheticData/train.csv').to_numpy()

def get_image_data():
    images_names = []
    output = []
    train_csv = pd.read_csv(f'{path_data}/SyntheticData/train.csv').to_numpy()
    for i, row in enumerate(train_csv) :
        img_name = row[0]
        images_names.append(img_name)
        output.append(row[1])
    images_names = np.array(images_names)
    output = np.array(output)
    return images_names, output

def get_images(images_names):
    images = []
    for img_name in images_names:
        img = cv2.imread(f'{path_data}/SyntheticData/images/{img_name}')
        if(img is None): continue
        resized_img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array / 255.0
        images.append(img_array)
    images = np.array(images)
    return images

class NeuralNetwork(nn.Module) : 
    def __init__(self, embedding_dim, context_dim, hidden_dim, vocab_size) : 
        super(NeuralNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=5)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg_pool = nn.AvgPool2d(kernel_size=3)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTMCell(input_size=embedding_dim + context_dim, hidden_size=hidden_dim)  # Modified for concatenation
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.model_no = 0
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def forward(self, x, y=None) : 
        max_len = y[0].size - 1
        batch_size = x.size(0)
        
        x = nn.functional.relu(self.conv1(x.float()))
        x = self.pool1(x)

        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)

        x = nn.functional.relu(self.conv4(x))
        x = self.pool4(x)

        x = nn.functional.relu(self.conv5(x))
        x = self.pool5(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        
        cnn_embedding = x


        h_t = cnn_embedding
        c_t = cnn_embedding

        start_token = np.zeros(batch_size)
        output = [self.embedding(torch.tensor(start_token, dtype=torch.long).to(device))]
        start_prob = np.zeros((batch_size, self.vocab_size))
        start_prob[:, 0] = 1000
        output_prob = [torch.tensor(start_prob, requires_grad=True, dtype = torch.float).to(device)]

        for i in range(max_len) :
            num = np.random.randint(2) 
            embedded_token = None
            if not (num == 0 or y is None) : 
                embedded_token = self.embedding(torch.tensor(y[:, i], dtype=torch.long).to(device))
            else : 
                embedded_token = torch.tensor(output[-1], dtype=torch.float).to(device)
            rnn_input = torch.cat((cnn_embedding, embedded_token), dim=-1)
            h_t, c_t = self.lstm(rnn_input, (h_t, c_t))
            distribution = self.fc(h_t)
            output.append(self.embedding(torch.argmax(distribution, dim=-1).to(device)))
            output_prob.append(distribution)
        output_prob = torch.stack(output_prob, dim =1)
        return output_prob
    
    def predict(self, x):  # Prediction for only 1, not batch

        x = nn.functional.relu(self.conv1(x.float()))
        x = self.pool1(x)

        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)

        x = nn.functional.relu(self.conv4(x))
        x = self.pool4(x)

        x = nn.functional.relu(self.conv5(x))
        x = self.pool5(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)

        cnn_embedding = x

        h_t = cnn_embedding
        c_t = cnn_embedding

        start_token = 0
        output = np.array([0])

        last_token = 0

        while output[-1] != 1:
            embedded_token = self.embedding(torch.tensor([output[-1]], dtype=torch.long).to(device))
            rnn_input = torch.cat((cnn_embedding, embedded_token), dim=-1)
            h_t, c_t = self.lstm(rnn_input, (h_t, c_t))
            distribution = self.fc(h_t)
            output.append(torch.argmax(distribution, dim=-1))
        return output[1:-1]

    def save_models(self):
        torch.save(self.state_dict(), f"./Fresh_Models/model_{self.model_no}.model")
        self.model_no += 1
        print(f"LSTM Checkpoint {self.model_no} saved")

    def load_model(self, load_model_no):
        self.load_state_dict(torch.load(f"./Fresh_Models/model_{load_model_no}.model"))
        print(f"LSTM Checkpoint {load_model_no} loaded")

def train(model, train_data_names, train_labels, vocab_size, epochs=10, batch_size=32, learning_rate=0.001, load_model_no=None) :
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if load_model_no is not None:
        model.load_model(load_model_no)
    for epoch in range(epochs) : 
        avg_loss = 0
        iteration = 0
        for i in range(0, len(train_data_names), batch_size) :
            optimizer.zero_grad()
            batch_data_names = train_data_names[i:i+batch_size]
            batch_data = get_images(batch_data_names)
            batch_labels = train_labels[i:i+batch_size]
            max_len = 0
            for j in range(len(batch_labels)) : 
                max_len = max(max_len, len(batch_labels[j]))
            for j in range(len(batch_labels)) : 
                padding_length = max_len - len(batch_labels[j])
                padding_values = np.full(padding_length, 2)
                batch_labels[j] = np.concatenate([np.array([0]), np.array(batch_labels[j]), np.array([1]), padding_values])
            batch_labels = np.vstack(batch_labels)
            batch_data = torch.tensor(batch_data, dtype=torch.float).to(device)
            output = model(batch_data, batch_labels)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float, requires_grad=True).to(device)
            reshape_output = output.reshape(-1, vocab_size)
            loss = criterion( reshape_output.to(device), batch_labels.reshape(-1).long().to(device))
            loss.backward()
            optimizer.step()
            print(f"{epoch} and {iteration} : {loss}")
            iteration += 1
            avg_loss += loss
        print(f"Epoch: {epoch} Loss: {avg_loss/iteration}")
        if(epoch % 10 == 0): model.save_models()

train_images_names, train_output = get_image_data()
vocab_dict = {}
inverse_dict = {}
vocab_dict['<START>'] = 0
inverse_dict[0] = '<START>'
vocab_dict['<END>'] = 1
inverse_dict[1] = '<END>'
vocab_dict['<PAD>'] = 2
inverse_dict[2] = '<PAD>'
for formula in train_output:
    tokens = formula.split()
    for word in tokens:
        if word not in vocab_dict:
            vocab_dict[word] = len(vocab_dict)
            inverse_dict[vocab_dict[word]] = word

vocab_size = len(vocab_dict)

train_output_new = []
for i in range(len(train_output)):
    tokens = train_output[i].split()
    for j in range(len(tokens)):
        tokens[j] = vocab_dict[tokens[j]]
    train_output_new.append(tokens)

for key in vocab_dict.keys():
    inverse_dict[vocab_dict[key]] = key

cnn_lstm = NeuralNetwork(512, 512, 512, vocab_size)
cnn_lstm.to(device)

train(cnn_lstm, train_images_names, train_output_new, vocab_size, 100, 250, 0.001, load_model_no=None)

def get_handwritten_data():
    # Load the images
    images_names = []
    output = []
    train_csv = pd.read_csv('./col_774_A4_2023/HandwrittenData/train_hw.csv').to_numpy()
    for i, row in enumerate(train_csv) :
        img_name = row[0]
        images_names.append(img_name)
        output.append(row[1])
    images_names = np.array(images_names)
    output = np.array(output)
    return images_names, output

def get_handwritten_images(images_names):
    images = []
    for img_name in images_names:
        img = cv2.imread('./col_774_A4_2023/HandwrittenData/images/train' + img_name)
        if(img is None): continue
        resized_img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array / 255.0
        images.append(img_array)
    images = np.array(images)
    return images

handwritten_images_names, handwritten_output = get_handwritten_data()

handwritten_output_new = []
for i in range(len(handwritten_output)):
    tokens = handwritten_output[i].split()
    for j in range(len(tokens)):
        if(tokens[j] in vocab_dict):
            tokens[j] = vocab_dict[tokens[j]]
        else:
            tokens[j] = np.random.uniform(0, vocab_size)
    handwritten_output_new.append(tokens)

handwritten_images = get_handwritten_images(handwritten_images_names)
train(cnn_lstm, handwritten_images, handwritten_output_new, vocab_size, 100, 64, 0.001, load_model_no=35)

def decode_string(prediction):
    output_string = ""
    for val in prediction:
        index = val.item()
        if index == 1 or index == 2: 
            break
        output_string += inverse_dict[index]
        output_string += " "
    return output_string

def get_prediction(model, input_img) :
    x = nn.functional.relu(model.conv1(input_img.float()))
    x = model.pool1(x)

    x = nn.functional.relu(model.conv2(x))
    x = model.pool2(x)

    x = nn.functional.relu(model.conv3(x))
    x = model.pool3(x)

    x = nn.functional.relu(model.conv4(x))
    x = model.pool4(x)

    x = nn.functional.relu(model.conv5(x))
    x = model.pool5(x)

    x = model.avg_pool(x)

    x = x.view(x.size(0), -1)

    cnn_embedding = x
    h_t = cnn_embedding
    c_t = cnn_embedding

    start_token = 0
    output = [0]

    last_token = 0

    while output[-1] != 1 and output[-1] != 2 and len(output) < 130:
        embedded_token = model.embedding(torch.tensor([output[-1]], dtype=torch.long).to(device))
        rnn_input = torch.cat((cnn_embedding, embedded_token), dim=-1)
        h_t, c_t = model.lstm(rnn_input, (h_t, c_t))
        distribution = model.fc(h_t)
        output.append(torch.argmax(distribution, dim=-1))
    return decode_string(output[1:])

def get_handwritten_test(images_names):
    images = []
    for img_name in images_names:
        img = cv2.imread(f'{path_data}/HandwrittenData/images/test/{img_name}')
        if(img is None): continue
        resized_img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array / 255.0
        images.append(img_array)
    images = np.array(images)
    return images

image_directory = f'{path_data}/HandwrittenData/images/test/'

image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

csv_file_path = 'pred1a.csv'

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image', 'formula'])
    for image_file in image_files:
        image0 = get_handwritten_test([image_file])
        input_img = torch.tensor(image0, dtype=torch.float).to(device)
        pred_string = get_prediction(cnn_lstm, input_img)
        print(image_file, pred_string)
        csv_writer.writerow([image_file, f'\'{pred_string}\''])

def get_synthetic_test(images_names):
    images = []
    for img_name in images_names:
        img = cv2.imread(f'{path_data}/SyntheticData/images/{img_name}')
        if(img is None): continue
        resized_img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_rgb)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array / 255.0
        images.append(img_array)
    images = np.array(images)
    return images

image_directory = f'{path_data}/SyntheticData/images/'

image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

csv_file_path = 'pred1b.csv'

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image', 'formula'])
    for image_file in image_files:
        image0 = get_synthetic_test([image_file])
        input_img = torch.tensor(image0, dtype=torch.float).to(device)
        pred_string = get_prediction(cnn_lstm, input_img)
        print(image_file, pred_string)
        csv_writer.writerow([image_file, f'\'{pred_string}\''])