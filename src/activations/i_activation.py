from numpy.typing import NDArray
from abc import ABC, abstractmethod
import numpy as np

class IActivation(ABC):
    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def derivative(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
