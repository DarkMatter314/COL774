# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# %%
points = 1000000
x1 = np.random.normal(3, 2, points)
x2 = np.random.normal(-1, 2, points)
e = np.random.normal(0, np.sqrt(2), points)
X = np.column_stack((np.ones(points), x1, x2))
Y = 3 + x1 + 2*x2 + e

# %%
def compute_error(X, Y, theta, start, end):
    m = X.shape[0]
    cost = 0
    for i in range(start-1, end):
        cost += (Y[i] - np.dot(theta, X[i]))**2
    cost = cost/(2*(end-start+1))
    return cost

# %%
def compute_gradient(X, Y, theta, start, end):
    m = X.shape[0]
    dj_dtheta = np.zeros(3)
    for i in range(start-1, end):
        dj_dtheta += (Y[i] - np.dot(theta, X[i]))*X[i]
    return dj_dtheta/(end-start+1)

# %%
m = X.shape[0]
theta = np.zeros(3)
compute_error(X, Y, theta, 1, m)

# %%
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# %%
def gradient_descent(X, Y, theta, max_iterations, learn_prm, batch_size, m, k, conv_crit):
    theta_hist = []
    sum = 0
    queue = []
    for i in range(max_iterations):
        batch_i = 1
        while batch_i*batch_size <= m:
            grad = np.array(compute_gradient(X, Y, theta, (batch_i-1)*batch_size + 1, batch_i*batch_size))
            error = compute_error(X, Y, theta, (batch_i-1)*batch_size + 1, batch_i*batch_size)
            theta_hist.append(np.array([theta[0], theta[1], theta[2]]))
            print(f"Iteration {batch_size*i + batch_i} | Theta: {theta} | Grad: {grad} | Sum: {sum})")
            if(batch_size*i + batch_i <= k): 
                queue.append(error)
                sum += error
            else:
                sum = sum + error - queue[0]
                queue.pop(0)
                queue.append(error)
                if(sum < k*conv_crit):
                    return (theta_hist, theta)
            theta = theta + learn_prm*grad
            batch_i += 1
    return (np.array(theta_hist), np.array(theta))

# %%
max_iterations = 100
learn_prm = 0.001
batch_size = 1
conv_crit = 0.1
k = 10
(theta_hist, theta_new) = gradient_descent(X, Y, theta, max_iterations, learn_prm, batch_size, m, k, conv_crit)
print(theta_new)

# %%
theta_new = np.array([2.73753315, 1.05221441, 1.98514639])

# %%
test_data = pd.read_csv("../data/q2/q2test.csv").to_numpy()
testX = test_data[:, 0:2]
ones_column = np.ones((testX.shape[0], 1))
testX = np.hstack((ones_column, testX))
testY = test_data[:, 2]
error_hyp = compute_error(testX, testY, theta_new, 1, testX.shape[0])
theta_actual = np.array([3, 1, 2])
error_act = compute_error(testX, testY, theta_actual, 1, testX.shape[0])
(error_hyp, error_act)

# %%
theta_hist_new = list(theta_hist)
theta_hist_new

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract θ values for each parameter (θ0, θ1, θ2)
theta0_vals = [theta[0] for theta in theta_hist]
theta1_vals = [theta[1] for theta in theta_hist]
theta2_vals = [theta[2] for theta in theta_hist]

# Plot the movement of θ in 3D space
ax.plot(theta0_vals, theta1_vals, theta2_vals)
ax.scatter(theta_new[0], theta_new[1], theta_new[2], c='red', label=f"Final θ: {theta_new}")

# Set labels for the axes
ax.set_xlabel('θ0')
ax.set_ylabel('θ1')
ax.set_zlabel('θ2')

# Add a legend
ax.legend()

# Set title and show the plot
plt.title('Movement of θ in 3D Space')
plt.show()


