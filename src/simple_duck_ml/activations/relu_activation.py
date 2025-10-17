from simple_duck_ml.activations.i_activation import IActivation
from numpy.typing import NDArray
import numpy as np

class ReLuActivation(IActivation):
    name = "relu"

    def __init__(self, max_value: float = 6.0) -> None:
        self.max_value = max_value

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(np.maximum(0, x), 0, self.max_value)

    def derivative(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x > 0).astype(np.float64)

