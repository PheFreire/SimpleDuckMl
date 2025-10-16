from numpy.typing import NDArray
from loss.i_loss import ILoss
import numpy as np

class MSELoss(ILoss):
    def __call__(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def derivative(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        y_true = y_true.reshape(y_pred.shape)
        return 2 * (y_pred - y_true) / y_true.size
    
