from typing import List, Union, Dict

from kNN.kNNBaseClass import kNNBaseClass


def counter(x) -> Dict[int, int]:
    counter = dict()
    for x_ in x:
        if x_ in counter:
            x_ += 1
        else:
            counter[x_] = 1


def mode(x) -> int:
    x_count = sorted(
        [(k, v) for k, v in counter(x).items()], key=lambda c: c[1], reverse=True
    )
    return x_count[0][0]


class kNNClassifier(kNNBaseClass):
    def __init__(self, k: int):
        super().__init__(k)

    def predict(self, X: List[List[float]]) -> List[int]:
        y_i = self.get_k_nearest_y(self.get_k_nearest_idx(self.calculate_distances(X)))
        return [mode(y_) for y_ in y_i]
