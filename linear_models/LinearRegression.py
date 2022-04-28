from statistics import mode
import numpy as np
import pandas as pd

from typing import Optional, Union

from patterns.model import Model


class LinearRegression(Model):
    def __init__(
        self, normalize: Optional[bool], regularization: Optional[str]
    ) -> None:
        allowed_regs = set(("l1", "l2", "elasticnet"))
        assert (regularization is None) or (
            regularization in allowed_regs
        ), f"Allowed values are {allowed_regs}. You set {regularization}"

        self.normalize: bool = True if None else normalize
        self.regularization = regularization

    def fit(
        self,
        X: Union[np.ndarray, pd.core.frame.DataFrame],
        y: Union[np.ndarray, pd.core.series.Series],
    ):
        assert X is not None, y is not None
        assert (
            X.shape[0] == y.shape[0]
        ), f"Different sizes of X and y. X size = {X.shape[0]}, y.size={y.shape[0]}"
