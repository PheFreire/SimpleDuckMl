from typing import Dict, Iterable, List, Optional, Self, Type
from simple_duck_ml.activations.i_activation import IActivation
from simple_duck_ml.layers.i_layer import ILayer
from numpy.typing import NDArray
import numpy as np

from simple_duck_ml.serializers.tensor_io import load_tensor, write_tensor
from simple_duck_ml.serializers.toml_io import load_toml, write_toml
from duckdi import Get
import uuid
import os

class ConvolutionalLayer(ILayer):
    name = "conv"

    def __init__(
        self,
        nodes_num: int,
        kernel_shape: Iterable[int],
        activation: IActivation,
        stride: int=1,
    ) -> None:
        self.nodes_num = nodes_num
        self.kernel_shape = list(kernel_shape)

        limit = np.sqrt(2.0 / np.prod(self.kernel_shape))
        self.w = np.random.randn(nodes_num, *self.kernel_shape) * limit
        self.b = np.zeros((nodes_num, 1))

        self._grad_w: NDArray[np.float64] = np.zeros_like(self.w)
        self._grad_b: NDArray[np.float64] = np.zeros_like(self.b)

        self.stride = stride
        self.activation = activation

        self._patches: Optional[NDArray[np.float64]] = None
        self._x: Optional[NDArray[np.float64]] = None
        self._output: Optional[NDArray[np.float64]] = None
    
    def __get_patches(self, x: NDArray[np.float64], stride: int=1) -> NDArray[np.float64]:
        # Expand shape to support data (only support 3D channels :/) 
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)

        kernel_h, kernel_w, in_channels = self.kernel_shape
        input_h, input_w, input_channels = x.shape

        # Validate channel dimensions
        if in_channels != input_channels:
            raise ValueError(
                f"Input channels ({input_channels}) must match kernel channels ({in_channels})."
            )

        # Compute output (activation map) shape
        output_h = (input_h - kernel_h) // stride + 1
        output_w = (input_w - kernel_w) // stride + 1

        # Allocate array for patches
        patches = np.zeros((output_h, output_w, kernel_h, kernel_w, in_channels))

        # Extract each patch (manual sliding window)
        for i, j in np.ndindex(output_h, output_w):
            start_i = i * stride
            start_j = j * stride
            patches[i, j] = x[start_i:start_i + kernel_h, start_j:start_j + kernel_w, :]

        return patches
            
    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)

        patches = self.__get_patches(x, self.stride)
        self._x = x
        self._patches = patches

        output_h, output_w = patches.shape[:2]
        kernel_h, kernel_w = self.kernel_shape[:2]
        in_channels = self.kernel_shape[-1]
        
        # Flat By Patch
        total_activation_map_elements_num = output_h * output_w
        total_kernel_elements_num = kernel_h * kernel_w * in_channels
        x_flat = patches.reshape(total_activation_map_elements_num, total_kernel_elements_num)

        # Flat Weights By Node
        w_flat = self.w.reshape(self.nodes_num, -1)

        # Forward Calc
        z = np.dot(w_flat, x_flat.T) + self.b
        
        # Resize to Activation Map Shape For Each Node
        z = z.reshape(self.nodes_num, output_h, output_w)
        
        self._output = self.activation(z)
    
        # Move the channel dimension from its current index ("0") to the last position ("-1")
        return np.moveaxis(self._output, 0, -1)

    def backward(self, delta: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._output is None or self._patches is None:
            raise RuntimeError("Forward should be called before Backward")

        output_h, output_w = self._patches.shape[:2]
        kernel_h, kernel_w = self.kernel_shape[:2]
        in_channels = self.kernel_shape[-1]

        # Move the channel dimension from its current index ("1") to the first position ("0")
        if delta.shape != self._output.shape:
            delta = np.moveaxis(delta, -1, 0)

        delta *= self.activation.derivative(self._output)

        # Flat By Patch
        total_activation_map_elements_num = output_h * output_w
        total_kernel_elements_num = kernel_h * kernel_w * in_channels
        x_flat = self._patches.reshape(total_activation_map_elements_num, total_kernel_elements_num)

        # Flat Weights By Node
        w_flat = self.w.reshape(self.nodes_num, -1)

        # Flat Delta by Node
        delta_flat = delta.reshape(self.nodes_num, total_activation_map_elements_num)

        # Calc and feed Weights Gradient Map 
        self._grad_w += np.dot(delta_flat, x_flat).reshape(self.w.shape)

        # Calc and feed Bias Gradient Map Ensuring The 2D Shape Of The Gradient Map
        self._grad_b += np.sum(delta, axis=(1, 2), keepdims=False).reshape(self._grad_b.shape)

        # Calc the x gradient
        grad_x_flat = np.dot(delta_flat.T, w_flat)
        
        # Re-shape the x gradient to kernel dimensions
        grad_x = np.zeros_like(self._x)
        for idx, (i, j) in enumerate(np.ndindex(output_h, output_w)):
            # np.ndindex is the same than:
                # for i in range(output_h): 
                #       for j in range(output_w):
            patch_grad = grad_x_flat[idx].reshape(kernel_h, kernel_w, in_channels)
            grad_x[i:i+kernel_h, j:j+kernel_w, :] += patch_grad

        self._patches = None
        return grad_x

    def clean_grad(self) -> None:
        self._grad_w.fill(0)
        self._grad_b.fill(0)
   
    def update(self, learning_rate: float = 0.01, batch_size: int = 1) -> None:
        self.w -= learning_rate * (self._grad_w / batch_size)
        self.b -= learning_rate * (self._grad_b / batch_size)

        self._x = None 
        self._patches = None
        
        if self._output is not None:
           self._output.fill(0)

        self.clean_grad()

    def save(self, name: Optional[str] = None, path: str = ".", overwrite: bool = True) -> str:
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
                "layer_type": self.name,
                "kernel_shape": self.kernel_shape,
                "activation": self.activation.name,
                "nodes_num": self.nodes_num,
                "stride": self.stride,
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
        
        kernel_shape = __process_keys(layer_info, "kernel_shape", List)
        activation = Get(
            IActivation, 
            label='activation',
            adapter=__process_keys(layer_info, "activation", str)
        )
        nodes_num = __process_keys(layer_info, "nodes_num", int)
        stride = __process_keys(layer_info, "stride", int)
        tensors_path = __process_keys(layer_info, "tensors_path", str)

        tensor_info = load_tensor(tensors_path, find_on_path=True)
        w = __process_keys(tensor_info, "w", np.ndarray)
        b = __process_keys(tensor_info, "b", np.ndarray)
        
        layer = cls(nodes_num, kernel_shape, activation, stride)

        layer.w = w
        layer.b = b
        return layer
