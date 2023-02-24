import os
import math
import numpy as np
from scipy.linalg import fractional_matrix_power


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Relu activation function
def relu(x):
    return np.maximum(0, x)


def gcn(a, h, w, diag):
    diagonal_half = fractional_matrix_power(diag, -0.5)
    eq = diagonal_half.dot(a).dot(diagonal_half).dot(h).dot(w)
    return relu(eq)


def setup(frame, vehicles, x, y, adj):
    diagonal_matrix = [[0 for i in range(len(vehicles))] for j in range(len(vehicles))]

    features = []

    # Combine x and y into one array
    for i in range(len(x)):
        features.append([x[i], y[i]])

    features = np.array(features)

    # Create the diagonal matrix
    for i in range(len(adj)):
        # Count 1 occurrences
        count = 0
        for j in range(len(vehicles)):
            if adj[i][j] == 1:
                count += 1
        diagonal_matrix[i][i] = count

    for i in range(len(adj)):
        for j in range(len(vehicles)):
            if adj[i][j] == 1:
                if i != j:
                    distance = math.sqrt(((x[j] - x[i]) ** 2) + (y[j] - y[i]) ** 2)
                    adj[i][j] = 1/distance

    # Initialize the weights
    np.random.seed(12345)
    n_h = 9     # Neurons in the hidden layer.
    n_y = 2     # Neurons in the output layer

    W0 = np.random.randn(features.shape[1], n_h) * 0.01
    W1 = np.random.randn(n_h, n_y) * 0.01

    hidden1 = gcn(adj, features, W0, diagonal_matrix)
    hidden2 = gcn(adj, hidden1, W1, diagonal_matrix)

    return hidden2
