from __future__ import annotations

from abc import ABC
from abc import abstractclassmethod

import numpy as np
import pandas as pd
from typing import Optional, Union


class ModelBase:
    def __init__(self) -> None:
        pass

    def fit(
        self,
    ) -> ModelBase:
        raise NotImplementedError

    def predict(self, X: Union[np.ndarray, pd.core.frame.DataFrame]):
        raise NotImplementedError

    def fit_predict(self, X: Union[np.ndarray, pd.core.frame.DataFrame]):
        self.fit()
        self.predict()
