from simple_duck_ml.loss.i_loss import ILoss
from numpy.typing import NDArray
import numpy as np

class MSELoss(ILoss):
    name = "mse"

    def __call__(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def derivative(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        y_true = y_true.reshape(y_pred.shape)
        return 2 * (y_pred - y_true) / y_true.size
    
