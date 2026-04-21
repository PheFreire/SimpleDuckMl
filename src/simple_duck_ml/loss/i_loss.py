from abc import ABC, abstractmethod
from duckdi import Interface
from numpy.typing import NDArray
import numpy as np

@Interface(label='loss')
class ILoss(ABC):
    name: str

    @abstractmethod
    def __call__(self, y_pred: NDArray[np.float32], y_true: NDArray[np.float32]) -> float: ...
    
    @abstractmethod
    def derivative(self, y_pred: NDArray[np.float32], y_true: NDArray[np.float32]) -> NDArray[np.float32]: ...

