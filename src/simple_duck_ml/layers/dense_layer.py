from numpy.typing import NDArray
from simple_duck_ml.activations.i_activation import IActivation
from simple_duck_ml.activations.softmax_activation import SoftmaxActivation
from simple_duck_ml.layers.i_layer import ILayer
from typing import Dict, Optional, Self, Type
import numpy as np

from simple_duck_ml.serializers.tensor_io import load_tensor, write_tensor
from simple_duck_ml.serializers.toml_io import load_toml, write_toml
from duckdi import Get
import uuid
import os

class DenseLayer(ILayer):
    name = "dense"

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

        if self.w is None or self.b is None or self._grad_w is None or self._grad_b is None:
            self._init_params(x.shape[0])

        if self.w is None:
          raise RuntimeError("Error: Could Not Initialize Dense Layer Weights") 

        z = np.dot(self.w, x) + self.b
        
        self._x = x
        self._z = z

        self._output = self.activation(z)
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

    def save(self, name: Optional[str] = None, path: str = ".", overwrite: bool=True) -> str:
        if self.w is None or self.b is None:
            raise RuntimeError("Error: Could Not Instance Dense Layer Weights/Bias") 

        name = str(uuid.uuid4()).replace("-", "") if name is None else name
        os.mkdir(os.path.join(path, name))
        file_path = os.path.join(path, name, name)

        tensors_path = write_tensor(
            tensors={ "w": self.w, "b": self.b },
            path=file_path,
            overwrite=overwrite
        )

        return write_toml(
            obj={
                "output_size": self.output_size,
                "input_size": self.input_size,
                "activation": self.activation.name,
                "layer_type": self.name,
                "tensors_path": tensors_path,
            },
            path=file_path,
            overwrite=overwrite,
        )
    
    @classmethod
    def load(cls, path: str) -> Self:
        layer_info = load_toml(path, find_on_path=True)

        def __process_keys[T](obj: Dict, key: str, expected_type: Type[T]) -> T:
            data = obj.get(key, None)
            if data is None or not isinstance(data, expected_type):
                raise KeyError(f"Error: Could Not Process \"{key}\" key!")

            return data
        
        output_size = __process_keys(layer_info, "output_size", int)
        activation = Get(
            IActivation,
            label='activation',
            adapter=__process_keys(layer_info, "activation", str),
        )
        input_size = __process_keys(layer_info, "input_size", int)
        tensors_path = __process_keys(layer_info, "tensors_path", str)

        tensor_info = load_tensor(tensors_path, find_on_path=True)
        w = __process_keys(tensor_info, "w", np.ndarray)
        b = __process_keys(tensor_info, "b", np.ndarray)
        
        layer = cls(output_size, activation)
        
        layer._init_params(input_size)
        layer.w = w
        layer.b = b

        return layer
