from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

class ILoss(ABC):
    @abstractmethod
    def __call__(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> float: ...
    
    @abstractmethod
    def derivative(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]: ...

