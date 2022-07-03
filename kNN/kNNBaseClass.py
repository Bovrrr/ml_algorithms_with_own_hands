from typing import List, Union

from patterns.model import ModelBase


def dot_product(
    x: List[Union[float, int]], y: List[Union[float, int]]
) -> List[Union[float, int]]:
    return [x[i] * y[i] for i in range(len(x))]


def L_p_distance(
    x: List[Union[float, int]],
    y: List[Union[float, int]],
    p: Union[float, int],
) -> Union[float, int]:
    return sum([(x[i] - y[i]) ** p for i in range(len(x))]) ** 1 / p


def argSort(x: List[Union[float, int]]) -> List[int]:
    x_i = list(zip(x, range(len(x))))
    x_i.sort(key=lambda c: c[0], reverse=True)
    return [c[1] for c in x_i]


class kNNBaseClass(ModelBase):
    def __init__(self, k: int):
        self.k = k
        self.is_fitted = False

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.d = len(X[0])
        self.is_fitted = True

    def calculate_distances(self, X):
        m, n = len(self.X), len(X)
        res = [[] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                res[i].append(L_p_distance(X[i], self.X[j], 2))
        return res

    def get_k_nearest_idx(self, distances):
        m = len(distances)
        res = [None for _ in range(m)]
        for i in range(m):
            res[i] = argSort(distances[i])[: self.k]
        return res

    def get_k_nearest_y(self, k_i):
        m = len(k_i)
        res = [None for _ in range(m)]
        for i in range(m):
            res[i] = [self.y[j] for j in k_i[i]]
        return res
