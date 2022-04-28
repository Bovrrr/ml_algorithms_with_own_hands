from __future__ import annotations

from abc import ABC
from abc import abstractclassmethod

import numpy as np
import pandas as pd
from typing import Optional, Union


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def fit(
        self,
    ) -> Model:
        raise NotImplementedError

    @abstractclassmethod
    def predict(self, X: Union(np.ndarray, pd.core.frame.DataFrame)):
        raise NotImplementedError

    # @abstractclassmethod
    # def predict_proba(self):
    #     raise NotImplementedError

    def fit_predict(self, X: Union(np.ndarray, pd.core.frame.DataFrame)):
        self.fit()
        self.predict()
