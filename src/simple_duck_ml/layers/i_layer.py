from abc import ABC, abstractmethod
from numpy.typing import NDArray
from duckdi import Interface
from typing import Optional, Self
import numpy as np

@Interface(label='layer')
class ILayer(ABC):
    name: str

    @abstractmethod
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    @abstractmethod
    def backward(self, delta: NDArray[np.float64]) -> NDArray[np.float64]: ...
    
    @abstractmethod
    def clean_grad(self) -> None: ...

    @abstractmethod
    def update(self, learning_rate: float = 0.01, batch_size: int = 1) -> None: ...

    @classmethod
    def load(cls, path: str) -> Self: ...

    @abstractmethod
    def save(self, name: Optional[str] = None, path: str = ".", overwrite: bool=True) -> str: ...
