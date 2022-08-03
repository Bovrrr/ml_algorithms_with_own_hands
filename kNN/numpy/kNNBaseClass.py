import numpy as np
import pandas as pd

from patterns.model import ModelBase


class kNNBaseClass(ModelBase):
    def __init__(self, k: int) -> None:
        self.is_fitted = False
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.d = X.shape[1]
        self.is_fitted = True

    def calculate_distances(self, X):
        distances = (
            np.sum(X**2, axis=1)[:, None]
            - 2 * X @ self.X.T
            + np.sum(self.X**2, axis=1)
        )
        return distances

    def get_k_nearest_idxs(self, distances):
        k_neares_idxs = np.zeros((distances.shape[0], self.k))
        for i in range(k_neares_idxs.shape[0]):
            k_neares_idxs[i] = np.argsort(distances[i])[: self.k]
        return k_neares_idxs.astype(int)

    def get_k_nearest_y(self, k_nearest_idxs):
        k_nearest_y = np.zeros(k_nearest_idxs.shape)
        for i in range(k_nearest_y.shape[0]):
            k_nearest_y[i] = self.y[k_nearest_idxs]
        return k_nearest_idxs
