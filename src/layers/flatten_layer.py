from layers.i_layer import ILayer
from numpy.typing import NDArray
import numpy as np


class FlattenLayer(ILayer):
    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] | None = None
        self._batch_size: int | None = None

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Receive (C, H, W, N) or (H, W, C) or (C, H, W) and flat into (features, N)
        """
        self._input_shape = x.shape
        
        # Batch (C, H, W, N)
        if x.ndim == 4:
            self._batch_size = x.shape[-1]
            flat = x.reshape(-1, self._batch_size)

        # Implicit Batch
        elif x.ndim == 3:
            self._batch_size = 1
            flat = x.reshape(-1, 1)
        
        # Already "(features, batch)"
        elif x.ndim == 2 and x.shape[1] > 1:    
            self._batch_size = x.shape[1]
            flat = x

        else:
            flat = x.reshape(-1, 1)
            self._batch_size = 1

        return flat


    def backward(self, delta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._input_shape is None:
            raise RuntimeError("Forward deve ser chamado antes do Backward")

        # Return into original Shape (C, H, W, N) or (C, H, W)
        return delta.reshape(self._input_shape)


    def clean_grad(self) -> None:
        self._input_shape = None
        self._batch_size = None


    def update(self, learning_rate: float = 0.01, batch_size: int = 1) -> None:
        pass
