from typing import Union

import numpy as np
import pandas as pd

from kNN.numpy.kNNBaseClass import kNNBaseClass




class kNNRegressor(kNNBaseClass):
    def __init__(
        self,
        k: int,
    ):
        super().__init__(k)

    def predict(self, X: Union[np.ndarray, pd.core.frame.DataFrame]) -> np.ndarray:
        ys_i = self.get_k_nearest_y(
            self.get_k_nearest_idxs(self.calculate_distances(X))
        )
        return np.array([np.mean(y_) for y_ in ys_i])
