"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        euc_dist = utils.euclidean_dist_squared(self.X, X_hat)
        R, C = X_hat.shape
        y_hat = np.zeros(R)

        for r in range(R):
            indices = np.argsort(euc_dist[:,r]) # sorts the indices based on the euc distance to r
            k_indicies = indices[:self.k];

            labels_of_k_indices = np.take(self.y, k_indicies); # labels associated with the closest neighbours
            y_hat[r] = utils.mode(labels_of_k_indices) # most common label to example

        return y_hat

            


