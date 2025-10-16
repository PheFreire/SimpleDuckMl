from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

class ILayer(ABC):
    @abstractmethod
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def backward(self, delta: NDArray[np.float64]) -> NDArray[np.float64]: ...
    
    @abstractmethod
    def clean_grad(self) -> None: ...

    @abstractmethod
    def update(self, learning_rate: float = 0.01, batch_size: int = 1) -> None: ...
