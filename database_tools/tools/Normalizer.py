import numpy as np


class Normalizer():
    def __init__(self):
        self._is_fit = False
        return

    def fit(self, X):
        self._min = np.min(X)
        self._max = np.max((X + np.abs(self._min)))
        self._is_fit = True

    def transform(self, X):
        if self._is_fit:
            X_ = (X + np.abs(self._min)) / self._max
            return X_
        else:
            raise AttributeError('Normalizer fit() method has not been called.')

    def fit_transform(self, X):
        self.fit(X)
        X_ = self.transform(X)
        return X_
