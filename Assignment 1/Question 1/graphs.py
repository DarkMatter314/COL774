import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

linearX = pd.read_csv("./data/q1/linearX.csv", header=None).to_numpy()
linearY = pd.read_csv("./data/q1/linearY.csv", header=None).to_numpy()

def normalize(X):
    mean = np.mean(X)
    stdev = np.std(X)
    # X = np.array([(value-mean)/stdev for value in X])
    X = (X-mean)/stdev
    return (X, mean, stdev)

(normX, meanX, stdevX) = normalize(linearX)
(normY, meanY, stdevY) = normalize(linearY)

ones_column = np.ones((normX.shape[0], 1))  # Create a column of ones
X = np.hstack((np.ones((normX.shape[0], 1)), normX.reshape(-1, 1)))  # Stack columns horizontally
Y = normY

def compute_error(X, Y, theta):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        cost += (Y[i] - np.dot(theta, X[i]))**2
    cost = cost/(2*m)
    return cost

def compute_gradient(X, Y, theta):
    m = X.shape[0]
    dj_dtheta = np.zeros(2)
    for i in range(m):
        dj_dtheta += (Y[i] - np.dot(theta, X[i]))*X[i]
    return dj_dtheta

def gradient_descent(X, Y, theta, iterations, learn_prm):
    theta_hist = []
    for i in range(iterations):
        grad = np.array(compute_gradient(X, Y, theta))
        error = compute_error(X, Y, theta)
        theta_hist.append([theta[0], theta[1]])
        print(f"Iteration {i} | Error {error} | Theta: {theta} | Grad: {grad}")
        theta = theta + learn_prm*grad
    return (theta_hist, theta)

m = X.shape
theta = np.zeros(2)
compute_error(X, Y, theta)
iterations = 100
learn_prm = 0.001
(theta_hist, theta_new) = gradient_descent(X, Y, theta, iterations, learn_prm)

# # Create a mesh grid for the parameter space
theta0_vals = np.linspace(-1, 1, 100)  # Adjust the range as needed for theta0
theta1_vals = np.linspace(-1, 1, 100)  # Adjust the range as needed for theta1
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# Calculate the error values for each parameter combination
error_vals = np.zeros_like(theta0_vals)
for i in range(len(theta0_vals)):
    for j in range(len(theta0_vals[i])):
        # Calculate error using the current parameters
        current_theta0 = theta0_vals[i, j]
        current_theta1 = theta1_vals[i, j]
        error_vals[i, j] = compute_error(X, Y, [current_theta0, current_theta1])  # Replace with your error calculation

# Create a contour plot of the error function
plt.contour(theta0_vals, theta1_vals, error_vals, levels=20, cmap='viridis')

scatter = plt.scatter([], [], color='red', label='Min Error Point')

def update(frame):
    thetanew = theta_hist[frame]
    scatter.set_offsets(np.array([[thetanew[0], thetanew[1]]]))
    return scatter,

# Create the animation
ani = FuncAnimation(plt.gcf(), update, frames=len(theta_hist), interval=200, blit=True)

plt.xlabel('Theta 0')
plt.ylabel('Theta 1')
plt.title('Error Function Contour Plot')
plt.colorbar()  # Add a colorbar to the plot
plt.legend()
plt.show()