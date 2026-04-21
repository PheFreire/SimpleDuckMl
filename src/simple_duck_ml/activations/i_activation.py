from numpy.typing import NDArray
from abc import ABC, abstractmethod
from duckdi import Interface
import numpy as np

@Interface(label='activation')
class IActivation(ABC):
    name: str

    @abstractmethod
    def __call__(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

    @abstractmethod
    def derivative(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...
