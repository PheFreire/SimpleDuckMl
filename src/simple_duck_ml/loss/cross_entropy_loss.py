from simple_duck_ml.loss.i_loss import ILoss
from numpy.typing import NDArray
import numpy as np

class CrossEntropyLoss(ILoss):
    name = 'cross_entropy'

    def __call__(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> float:
        if y_true.shape != y_pred.shape:
            y_true = y_true.reshape(y_pred.shape)

        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def derivative(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        return y_pred - y_true

