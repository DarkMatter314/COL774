# %%
'''
This is started code for part a. 
Using this code is OPTIONAL and you may write code from scratch if you want
'''


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# %%
label_encoder = None 

def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OrdinalEncoder()
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

# %%
class DTNode:

    def __init__(self, depth, is_leaf = False, value = 0, column = None, split_values = [], thresh = 0.0, type = 0):
        #to split on column
        self.depth = depth

        #add children afterwards
        self.children = []

        #if leaf then also need value
        self.is_leaf = is_leaf
        if(self.is_leaf):
            self.value = value
        
        if(not self.is_leaf):
            self.value = value
            self.column = column
            # split_values is a list denoting the values on which to split on children
            self.split_values = split_values
            self.thresh = thresh
            self.type = type

    def add_child(self, child):
        self.children.append(child)

    def get_children(self, X):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
        '''
        if(self.type == 0):
            for i in range(len(self.split_values)):
                if(X[self.column] == self.split_values[i]):
                    return self.children[i]
            return None
        else:
            if(X[self.column] <= self.thresh):
                return self.children[0]
            else:
                return self.children[1]


# %%
class DTTree:

    def __init__(self):
        #Tree root should be DTNode
        self.root = None

    def entropy(self, y):
        '''
        Return entropy of y
        Args:
            y: numpy array of shape [num_samples, 1]
        Returns:
            entropy: scalar value
        '''
        y_zeros = np.count_nonzero(y == 0)
        total_y = y.shape[0]
        if(y_zeros == total_y or y_zeros == 0):
            return 0
        else:
            return -y_zeros/total_y*np.log2(y_zeros/total_y) - (total_y-y_zeros)/total_y*np.log2((total_y-y_zeros)/total_y)
        
    def get_best_attribute(self, X, y, types):
        entropy_i = self.entropy(y)
        max_gain = 0
        best_attribute = 0
        for i in range(X.shape[1]):
            new_entropy = 0
            if(types[i] == 'cont'):
                unique_values = np.unique(X[:,i])
                unique_values = np.sort(unique_values)
                # set threshold to median of unique values
                thresh = np.median(unique_values)
                lower_indices = np.where(X[:,i] <= thresh)
                lower_total = y[lower_indices].shape[0]
                upper_indices = np.where(X[:,i] > thresh)
                upper_total = y[upper_indices].shape[0]
                new_entropy = lower_total/X.shape[0]*self.entropy(y[lower_indices]) + upper_total/X.shape[0]*self.entropy(y[upper_indices])
            else:
                unique_values = np.unique(X[:,i])
                for j in range(unique_values.shape[0]):
                    indices = np.where(X[:,i] == unique_values[j])
                    new_entropy += indices[0].shape[0]/X.shape[0]*self.entropy(y[indices])
            new_gain = entropy_i - new_entropy
            if(new_gain > max_gain):
                max_gain = new_gain
                best_attribute = i
        return best_attribute

    def train(self, X, y, types, depth, max_depth):
        ''' 
        Return a node of class DTNode
        '''   
        y_zeros = np.count_nonzero(y == 0)
        total_y = y.shape[0]
        if(y_zeros == total_y):
            return DTNode(depth, True, 0, None, [], 0, 0)
        if(y_zeros == 0):
            return DTNode(depth, True, 1, None, [], 0, 0)
        if(depth == max_depth):
            if(y_zeros >= total_y/2):
                return DTNode(depth, True, 0, None, [], 0, 0)
            else:
                return DTNode(depth, True, 1, None, [], 0, 0)
        best_attribute = self.get_best_attribute(X, y, types)
        if(types[best_attribute] == 'cont'):
            unique_values = np.unique(X[:,best_attribute])
            unique_values = np.sort(unique_values)
            # set threshold to median of unique values
            thresh = np.median(unique_values)
            lower_indices = np.where(X[:,best_attribute] <= thresh)
            upper_indices = np.where(X[:,best_attribute] > thresh)
            y_zeros = np.count_nonzero(y[lower_indices] == 0)
            total_y = y[lower_indices].shape[0]
            value = 0 if y_zeros >= total_y/2 else 1
            node = DTNode(depth, False, value, best_attribute, [], float(thresh), 1)
            node.add_child(self.train(X[lower_indices], y[lower_indices], types, depth+1, max_depth))
            node.add_child(self.train(X[upper_indices], y[upper_indices], types, depth+1, max_depth))
            return node
        else:
            unique_values = np.unique(X[:,best_attribute])
            y_zeros = np.count_nonzero(y == 0)
            total_y = y.shape[0]
            value = 0 if y_zeros >= total_y/2 else 1
            node = DTNode(depth, False, value, best_attribute, unique_values, 0, 0)
            for i in range(unique_values.shape[0]):
                indices = np.where(X[:,best_attribute] == unique_values[i])
                node.add_child(self.train(X[indices], y[indices], types, depth+1, max_depth))
            return node
        

    def fit(self, X, y, types, max_depth = 10):
        '''
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            None
        '''
        self.root = self.train(X, y, types, 0, max_depth)

    def predict(self, node, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        if(node.is_leaf):
            return node.value
        else:
            child = node.get_children(X)
            if (child is not None): return self.predict(child, X)
            else: return node.value

    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        
    
    def post_prune(self, X_val, y_val):
        #TODO
        pass

# %%
X_train,y_train = get_np_array('train.csv')
X_test, y_test = get_np_array("test.csv")

types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]

# %%
depths = [5, 10, 15, 20, 25, 30, 40, 50]
trainfile = open("train_a.txt", "w")
testfile = open("test_a.txt", "w")

for depth in depths:
    tree = DTTree()
    tree.fit(X_train, y_train, types, depth)
    # print(f"Training Complete for depth {depth}")
    # training accuracy
    train_correct = 0
    train_incorrect = 0
    for i in range(X_train.shape[0]):
        if(tree.predict(tree.root, X_train[i]) == y_train[i]): train_correct += 1
        else: train_incorrect += 1
    # print(f"Training Accuracies:")
    # print(f"Correct: {train_correct} | Incorrect: {train_incorrect} | Accuracy: {train_correct/(train_correct+train_incorrect)}")
    trainfile.write(f"({depth}, {train_correct/(train_correct+train_incorrect)})\n")
    # testing accuracy
    test_correct = 0
    test_incorrect = 0
    for i in range(X_test.shape[0]):
        if(tree.predict(tree.root, X_test[i]) == y_test[i]): test_correct += 1
        else: test_incorrect += 1
    # print(f"Testing Accuracies:")
    # print(f"Correct: {test_correct} | Incorrect: {test_incorrect} | Accuracy: {test_correct/(test_correct+test_incorrect)}")
    # print()
    testfile.write(f"({depth}, {test_correct/(test_correct+test_incorrect)})\n")
trainfile.close()
testfile.close()

# %%
y_train_zeros = np.count_nonzero(y_train == 0)
y_train_total = y_train.shape[0]
y_train_ones = y_train_total - y_train_zeros
y_test_zeros = np.count_nonzero(y_test == 0)
y_test_total = y_test.shape[0]
y_test_ones = y_test_total - y_test_zeros
print(f"Training Data: {y_train_zeros} | {y_train_ones} | {y_train_zeros/y_train_total} | {y_train_ones/y_train_total}")
print(f"Testing Data: {y_test_zeros} | {y_test_ones} | {y_test_zeros/y_test_total} | {y_test_ones/y_test_total}")


