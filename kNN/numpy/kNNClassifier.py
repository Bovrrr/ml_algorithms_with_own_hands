from typing import Union

import numpy as np
import pandas as pd

from kNN.numpy.kNNBaseClass import kNNBaseClass


def mode(y_s):
    values, counts = np.unique(y_s, return_counts=True)
    return values[np.argmax(counts)]


class kNNClassifier(kNNBaseClass):
    def __init__(self, k: int) -> None:
        super().__init__(k)

    def predict(self, X: Union[np.ndarray, pd.core.frame.DataFrame]) -> np.ndarray:
        ys_i = self.get_k_nearest_y(
            self.get_k_nearest_idxs(self.calculate_distances(X))
        )
        return np.array([mode(y_) for y_ in ys_i])
