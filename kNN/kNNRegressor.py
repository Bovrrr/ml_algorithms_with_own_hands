from typing import List, Union

from kNN.kNNBaseClass import kNNBaseClass


def mean(x) -> float:
    return sum(x) / len(x)


class kNNRegressor(kNNBaseClass):
    def __init__(self, k: int) -> None:
        super().__init__(k)

    def predict(self, X: List[List[float]]) -> List[float]:
        y_i = self.get_k_nearest_y(self.get_k_nearest_idx(self.calculate_distances(X)))
        return [mean(y_) for y_ in y_i]
