import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import count_lc
from scipy.linalg import fractional_matrix_power


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def graph_convolution(frame, vehicles, x, y, adj):
    diagonal_matrix = [[0 for i in range(len(vehicles))] for j in range(len(vehicles))]

    features = []

    # Combine x and y into one array
    for i in range(len(x)):
        features.append([x[i], y[i]])

    print(features)

    # Create the diagonal matrix
    for i in range(len(adj)):
        # Count 1 occurrences
        count = 0
        for j in range(len(vehicles)):
            if adj[i][j] == 1:
                count += 1
        diagonal_matrix[i][i] = count

    # D^(-1/2)
    diagonal_half = fractional_matrix_power(diagonal_matrix, -0.5)
    DADX = diagonal_half.dot(adj).dot(diagonal_half).dot(features)
    print('DADX:\n', DADX)

    return frame

