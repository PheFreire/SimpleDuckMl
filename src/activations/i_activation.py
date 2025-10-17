from duckdi import Interface
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import numpy as np

@Interface(label='activation')
class IActivation(ABC):
    name: str

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def derivative(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
