from numpy.typing import NDArray
from activations.i_activation import IActivation
from activations.softmax_activation import SoftmaxActivation
from layers.i_layer import ILayer
from typing import Optional
import numpy as np


class DenseLayer(ILayer):
    def __init__(self, output_size: int, activation: IActivation) -> None:
        self.output_size = output_size
        self.activation = activation

        self.input_size: Optional[int] = None
        self.w: Optional[NDArray[np.float64]] = None
        self.b: Optional[NDArray[np.float64]] = None
        self._x: Optional[NDArray[np.float64]] = None
        self._z: Optional[NDArray[np.float64]] = None
        self._output: Optional[NDArray[np.float64]] = None

        self._grad_w: Optional[NDArray[np.float64]] = None
        self._grad_b: Optional[NDArray[np.float64]] = None


    def _init_params(self, input_size: int) -> None:
        self.input_size = input_size

        limit = np.sqrt(2.0 / self.input_size)
        self.w = np.random.randn(self.output_size, self.input_size) * limit
        self.b = np.zeros((self.output_size, 1))
        self._grad_w = np.zeros_like(self.w)
        self._grad_b = np.zeros_like(self.b)


    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if self.w is None or self.b is None:
            self._init_params(x.shape[0])

        if self.w is None:
          raise RuntimeError("Error: Could Not Initialize Dense Layer Weights") 

        z = np.dot(self.w, x) + self.b
        
        self._x = x
        self._z = z
        self._output = self.activation(z)

        print(f"[Dense] z.mean={z.mean():.4f}, z.std={z.std():.4f}, out.mean={self._output.mean():.4f}")
        return self._output


    def backward(self, delta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._output is None or self._x is None or self._z is None:
            raise RuntimeError("Forward should be called before Backward")

        if self.w is None:
          raise RuntimeError("Error: Could Not Initialize Dense Layer Weights") 

        # Ensure (classes, batch) shape
        delta = delta.reshape(self.output_size, -1)

        if not isinstance(self.activation, SoftmaxActivation):
            delta *= self.activation.derivative(self._z)

        batch_size = delta.shape[1]
        self._grad_w += np.dot(delta, self._x.T) / batch_size
        self._grad_b += np.mean(delta, axis=1, keepdims=True)

        d_x = np.dot(self.w.T, delta)
        return d_x


    def update(self, learning_rate: float = 0.01, batch_size: int = 1) -> None:
        if self._grad_w is None or self._grad_b is None:
            return

        if self.w is None or self.b is None:
          raise RuntimeError("Error: Could Not Initialize Dense Layer Weights/Bias") 

        self.w -= learning_rate * (self._grad_w / batch_size)
        self.b -= learning_rate * (self._grad_b / batch_size)

        self.clean_grad()
        self._x = None
        self._output = None
        self._z = None


    def clean_grad(self) -> None:
        if self._grad_w is not None:
            self._grad_w.fill(0)

        if self._grad_b is not None:
           self._grad_b = np.zeros_like(self.b)

