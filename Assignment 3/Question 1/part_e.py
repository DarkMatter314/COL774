# %%
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

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
X_train, y_train = get_np_array("train.csv")
X_test, y_test = get_np_array("test.csv")
X_val, y_val = get_np_array("val.csv")

# %%
X_train[1]

# %%
y_train = y_train.ravel()
y_test = y_test.ravel()
y_val = y_val.ravel()

# %%
# Varying max-depth
depths = [15, 25, 35, 45]
opfile = open('part_d.txt', 'w')
for depth in depths:
    classifier = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
    classifier.fit(X_train, y_train)
    predicted_y_train = classifier.predict(X_train)
    predicted_y_val = classifier.predict(X_val)
    predicted_y_test = classifier.predict(X_test)
    correct_train = np.count_nonzero(predicted_y_train == y_train)
    incorrect_train = np.count_nonzero(predicted_y_train != y_train)
    correct_test = np.count_nonzero(predicted_y_test == y_test)
    incorrect_test = np.count_nonzero(predicted_y_test != y_test)
    correct_val = np.count_nonzero(predicted_y_val == y_val)
    incorrect_val = np.count_nonzero(predicted_y_val != y_val)
    opfile.write(f"Max depth: {depth}\n")
    opfile.write(f"Training:   Correct: {correct_train} | Incorrect = {incorrect_train} | Accuracy = {correct_train / (correct_train + incorrect_train)}\n")
    opfile.write(f"Testing:    Correct: {correct_test} | Incorrect = {incorrect_test} | Accuracy = {correct_test / (correct_test + incorrect_test)}\n")
    opfile.write(f"Validation: Correct: {correct_val} | Incorrect = {incorrect_val} | Accuracy = {correct_val / (correct_val + incorrect_val)}\n\n")
opfile.close()

# %%
ccp_alpha_values = [0.001, 0.01, 0.1, 0.2]
opfile = open('part_d_ccp.txt', 'a')
for ccp_alpha in ccp_alpha_values:
    classifier = DecisionTreeClassifier(criterion='entropy', ccp_alpha=ccp_alpha)
    classifier.fit(X_train, y_train)
    predicted_y_train = classifier.predict(X_train)
    predicted_y_val = classifier.predict(X_val)
    predicted_y_test = classifier.predict(X_test)
    correct_train = np.count_nonzero(predicted_y_train == y_train)
    incorrect_train = np.count_nonzero(predicted_y_train != y_train)
    correct_test = np.count_nonzero(predicted_y_test == y_test)
    incorrect_test = np.count_nonzero(predicted_y_test != y_test)
    correct_val = np.count_nonzero(predicted_y_val == y_val)
    incorrect_val = np.count_nonzero(predicted_y_val != y_val)
    opfile.write(f"CCP Alpha: {ccp_alpha}\n")
    opfile.write(f"Training:   Correct: {correct_train} | Incorrect = {incorrect_train} | Accuracy = {correct_train / (correct_train + incorrect_train)}\n")
    opfile.write(f"Testing:    Correct: {correct_test} | Incorrect = {incorrect_test} | Accuracy = {correct_test / (correct_test + incorrect_test)}\n")
    opfile.write(f"Validation: Correct: {correct_val} | Incorrect = {incorrect_val} | Accuracy = {correct_val / (correct_val + incorrect_val)}\n\n")
opfile.close()

# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

# %%
param_grid = {
    'n_estimators': [50, 150, 250, 350],
    'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'min_samples_split': [2, 4, 6, 8, 10]
}

# %%
best_accuracy = 0
best_params = None
all_scores = open('part_d_all_rf.txt', 'w')
i = 0
# loop over all possible combinations of param_grid
for g in ParameterGrid(param_grid):
    print(i)
    # for each combination, train a RandomForestClassifier
    rf = RandomForestClassifier(oob_score=True, random_state=0, **g)
    rf.fit(X_train, y_train)
    oob_accuracy = rf.oob_score_
    y_pred = rf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    all_scores.write(f"Parameters: {g} | OOB accuracy: {100 * oob_accuracy} | Validation Accuracy: {100 * val_accuracy}\n")
    if(val_accuracy > best_accuracy):
        best_accuracy = val_accuracy
        best_params = g
    i += 1


# %%
all_scores.close()

# %%
best_rf = RandomForestClassifier(oob_score=True, random_state=0, **best_params)
best_rf.fit(X_train, y_train)

# %%
y_train_pred = best_rf.predict(X_train)
y_val_pred = best_rf.predict(X_val)
y_test_pred = best_rf.predict(X_test)
best_rf = open('part_d_best_rf.txt', 'w')
best_rf.write(f"Best parameters: {best_params}\n")
best_rf.write(f"Training accuracy: {accuracy_score(y_train, y_train_pred)}\n")
best_rf.write(f"Validation accuracy: {accuracy_score(y_val, y_val_pred)}\n")
best_rf.write(f"Testing accuracy: {accuracy_score(y_test, y_test_pred)}\n")

# %%
best_rf.close()


