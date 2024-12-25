# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%
logisticX = pd.read_csv('../data/q3/logisticX.csv', header=None).to_numpy()
logisticY = pd.read_csv('../data/q3/logisticY.csv', header=None).to_numpy()

# %%
def normalize(X):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    X = (X-mean)/stdev
    return (X, mean, stdev)

# %%
(normX, meanX, stdevX) = normalize(logisticX)
Y = logisticY
X = np.hstack((np.ones((normX.shape[0], 1)), normX))

# %%
def signmoid(X, theta):
    return 1/(1+np.exp(-np.dot(X, theta)))

# %%
def newton(X, theta, Y, max_iterations):
    theta_hist = []
    for i in range(max_iterations):
        theta_hist.append(np.array(theta))
        h = signmoid(X, theta)
        D = np.diag((h*(1-h)))
        H = X.T @ D @ X
        h_mat = np.array([h]).T
        H_inv = np.linalg.inv(H)
        grad = X.T@((h_mat)-Y)
        theta = theta - (H_inv@grad).reshape(theta.shape)
    return (theta, theta_hist)

# %%
theta = np.zeros(X.shape[1])
(theta_new, theta_hist) = newton(X, theta, Y, 10)
theta_hist

# %%
Y.shape

# %%
Y

# %%
X1 = X[Y[:, 0] == 1]
X0 = X[Y[:, 0] == 0]
X1

# %%
plt.scatter(X1[:, 1], X1[:, 2], marker='s', color='blue', label='Data Points with Y=1')
plt.scatter(X0[:, 1], X0[:, 2], marker='^', color='green', label='Data Points with Y=0')
x_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
y_pred = theta_new[0] + theta_new[1]*x_range
plt.plot(x_range, y_pred, color='red', label='Hypothesis Function')
plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.5)  # Vertical line at x=0
# Set appropriate axis limits
x_min = min(X[:, 1])
x_max = max(X[:, 1])
y_min = min(Y[:, 0])
y_max = max(Y[:, 0])
# y_center = 1  # Center of y-axis values
# Set x-axis and y-axis limits based on data ranges and centering of y-axis
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Raw Data and Hypothesis Function')
plt.legend()
plt.show()


