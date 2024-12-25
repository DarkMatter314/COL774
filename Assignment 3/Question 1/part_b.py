# %%
'''
This is started code for part b and c. 
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
        label_encoder = OneHotEncoder(sparse_output = False)
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
        self.train_accuracy = []
        self.val_accuracy = []
        self.test_accuracy = []

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
        pass

    def get_nodes(self, node, all_nodes):
        if(node.is_leaf):
            all_nodes.append(node)
            return
        else:
            all_nodes.append(node)
            for child in node.children:
                self.get_nodes(child, all_nodes)
            return
    
    def post_prune(self, X_val, y_val, consecutive_constant = 0, X_train = [], y_train = [], X_test = [], y_test = []):
        '''
        Post prune the tree
        Args:
            X_val: numpy array of data [num_samples, num_features]
            y_val: numpy array of classes [num_samples, 1]
        Returns:
            None
        '''
        correct = 0
        incorrect = 0
        all_nodes = []
        self.get_nodes(self.root, all_nodes)
        best_node_to_prune = None
        for i in range(X_val.shape[0]):
            if(self.predict(self.root, X_val[i]) == y_val[i]): correct += 1
            else: incorrect += 1
        val_accuracy = correct/(correct+incorrect)
        best_accuracy = correct/(correct+incorrect)
        for node in all_nodes:
            if(node.is_leaf): continue
            node.is_leaf = True
            correct = 0
            incorrect = 0
            for i in range(X_val.shape[0]):
                if(self.predict(self.root, X_val[i]) == y_val[i]): correct += 1
                else: incorrect += 1
            accuracy = correct/(correct+incorrect)
            if(accuracy >= best_accuracy):
                best_accuracy = accuracy
                best_node_to_prune = node
            node.is_leaf = False
        if(best_node_to_prune is not None):
            if(best_accuracy == val_accuracy): consecutive_constant += 1
            else: consecutive_constant = 0
            if(consecutive_constant == 5): return
            best_node_to_prune.is_leaf = True
            new_nodes =[]
            self.get_nodes(self.root, new_nodes)
            num_nodes = len(new_nodes)
            train_correct = 0
            train_incorrect = 0
            for i in range(X_train.shape[0]):
                if(tree.predict(tree.root, X_train[i]) == y_train[i]): train_correct += 1
                else: train_incorrect += 1
            self.train_accuracy.append([num_nodes, train_correct/(train_correct+train_incorrect)])
            test_correct = 0
            test_incorrect = 0
            for i in range(X_test.shape[0]):
                if(tree.predict(tree.root, X_test[i]) == y_test[i]): test_correct += 1
                else: test_incorrect += 1
            self.test_accuracy.append([num_nodes, test_correct/(test_correct+test_incorrect)])
            self.val_accuracy.append([num_nodes, best_accuracy])
            self.post_prune(X_val, y_val, consecutive_constant, X_train, y_train, X_test, y_test)
        return


# %%
#change the path if you want
X_train,y_train = get_np_array('train.csv')
X_test, y_test = get_np_array("test.csv")

#only needed in part (c)
X_val, y_val = get_np_array("val.csv")

types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]
while(len(types) != X_train.shape[1]):
    types = ['cat'] + types

# max_depth = 10
# tree = DTTree()
# tree.fit(X_train,y_train,types, max_depth = max_depth)

# %%
depths = [5, 15, 25, 35, 45, 55, 65]
# opfile = open("output.txt", "w")
trainfile = open("train_b.txt", "w")
testfile = open("test_b.txt", "w")

for depth in depths:
    tree = DTTree()
    tree.fit(X_train, y_train, types, depth)
    # opfile.write(f"Training Complete for depth {depth}\n")
    # training accuracy
    train_correct = 0
    train_incorrect = 0
    for i in range(X_train.shape[0]):
        if(tree.predict(tree.root, X_train[i]) == y_train[i]): train_correct += 1
        else: train_incorrect += 1
    trainfile.write(f"({depth}, {train_correct/(train_correct+train_incorrect)})\n")
    # opfile.write(f"Training Accuracies:\n")
    # opfile.write(f"Correct: {train_correct} | Incorrect: {train_incorrect} | Accuracy: {train_correct/(train_correct+train_incorrect)}\n")
    # testing accuracy
    test_correct = 0
    test_incorrect = 0
    for i in range(X_test.shape[0]):
        if(tree.predict(tree.root, X_test[i]) == y_test[i]): test_correct += 1
        else: test_incorrect += 1
    testfile.write(f"({depth}, {test_correct/(test_correct+test_incorrect)})\n")
    # opfile.write(f"Testing Accuracies:\n")
    # opfile.write(f"Correct: {test_correct} | Incorrect: {test_incorrect} | Accuracy: {test_correct/(test_correct+test_incorrect)}\n")
# opfile.close()
testfile.close()
trainfile.close()

# %%
depths = [45]
opfile = open("output_45.txt", "w")
trainfile = open("train_c_45.txt", "w")
testfile = open("test_c_45.txt", "w")
valfile = open("val_c_45.txt", "w")
node_train_files = []
node_test_files = []
node_val_files = []
for depth in depths:
    node_train_files.append(open(f"nodes_train_{depth}.txt", "w"))
    node_test_files.append(open(f"nodes_test_{depth}.txt", "w"))
    node_val_files.append(open(f"nodes_val_{depth}.txt", "w"))

for k in range(len(depths)):
    tree = DTTree()
    depth = depths[k]
    tree.fit(X_train, y_train, types, depth)
    tree.post_prune(X_val, y_val, 0, X_train, y_train, X_test, y_test)
    print(f"Trained and pruned for depth: {depth}")
    opfile.write(f"Training Complete for depth {depth}\n")
    # training accuracy
    train_correct = 0
    train_incorrect = 0
    for i in range(X_train.shape[0]):
        if(tree.predict(tree.root, X_train[i]) == y_train[i]): train_correct += 1
        else: train_incorrect += 1
    trainfile.write(f"({depth}, {train_correct/(train_correct+train_incorrect)})\n")
    opfile.write(f"Training Accuracies:\n")
    opfile.write(f"Correct: {train_correct} | Incorrect: {train_incorrect} | Accuracy: {train_correct/(train_correct+train_incorrect)}\n")
    # testing accuracy
    test_correct = 0
    test_incorrect = 0
    for i in range(X_test.shape[0]):
        if(tree.predict(tree.root, X_test[i]) == y_test[i]): test_correct += 1
        else: test_incorrect += 1
    testfile.write(f"({depth}, {test_correct/(test_correct+test_incorrect)})\n")
    opfile.write(f"Testing Accuracies:\n")
    opfile.write(f"Correct: {test_correct} | Incorrect: {test_incorrect} | Accuracy: {test_correct/(test_correct+test_incorrect)}\n")
    # validation accuracy
    val_correct = 0
    val_incorrect = 0
    for i in range(X_val.shape[0]):
        if(tree.predict(tree.root, X_val[i]) == y_val[i]): val_correct += 1
        else: val_incorrect += 1
    valfile.write(f"({depth}, {val_correct/(val_correct+val_incorrect)})\n")
    opfile.write(f"Validation Accuracies:\n")
    opfile.write(f"Correct: {val_correct} | Incorrect: {val_incorrect} | Accuracy: {val_correct/(val_correct+val_incorrect)}\n\n")
    # nodes
    for j in range(len(tree.train_accuracy)):
        node_train_files[k].write(f"({tree.train_accuracy[j][0]}, {100 * tree.train_accuracy[j][1]})\n")
        node_test_files[k].write(f"({tree.test_accuracy[j][0]}, {100 * tree.test_accuracy[j][1]})\n")
        node_val_files[k].write(f"({tree.val_accuracy[j][0]}, {100 * tree.val_accuracy[j][1]})\n")
    node_train_files[k].close()
    node_test_files[k].close()
    node_val_files[k].close()
opfile.close()
testfile.close()
trainfile.close()
valfile.close()

# %%
opfile.close()
testfile.close()
trainfile.close()
valfile.close()


