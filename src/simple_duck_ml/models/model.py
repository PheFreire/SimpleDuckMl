from simple_duck_ml.serializers.toml_io import load_toml, write_toml
from simple_duck_ml.dataset_unpacker.dataset import Dataset
from typing import Dict, List, Optional, Self, Type
from simple_duck_ml.layers.i_layer import ILayer
from simple_duck_ml.loss.i_loss import ILoss
from numpy.typing import NDArray
from tqdm import trange
from duckdi import Get
import numpy as np
import uuid
import os

class Model:
    def __init__(
        self,
        layers: List[ILayer],
        loss: ILoss,
        learning_rate: float
    ) -> None:
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> None:
        delta = self.loss.derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self, batch_size: int = 1) -> None:
        for layer in self.layers:
            layer.update(self.learning_rate, batch_size)

    def _one_hot(self, label: int, num_classes: int) -> NDArray[np.float64]:
        one_hot = np.zeros((num_classes, 1), dtype=np.float64)
        one_hot[label, 0] = 1.0
        return one_hot

    def fit(
        self,
        datasets: List[Dataset],
        epochs: int = 5,
        batch_size: int = 10,
        shuffle: bool = True,
        verbose: bool = True
    ) -> None:
        num_classes = len(datasets)
        samples = [(img, d.y) for d in datasets for img in d.x]
        total_samples = len(samples)

        for epoch in trange(epochs, desc="Epoch"):
            if shuffle:
                np.random.shuffle(samples)

            total_loss = 0.0

            # Mini batch loop
            for batch_start in range(0, total_samples, batch_size):
                batch = samples[batch_start:batch_start + batch_size]
                batch_loss = 0.0

                for x, label in batch:
                    y_true = self._one_hot(label, num_classes)
                    y_pred = self.forward(x)

                    loss = self.loss(y_pred, y_true)
                    batch_loss += float(loss)
                    self.backward(y_pred, y_true)

                self.update(batch_size)
                total_loss += batch_loss

            avg_loss = total_loss / total_samples
            if verbose:
                print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")
                print("-=" * 30)

    def save(self, name: Optional[str] = None, path: str = ".", overwrite: bool = True) -> str:
        name = str(uuid.uuid4()).replace("-", "") if name is None else name
        model_dir = os.path.join(path, name)

        os.makedirs(model_dir, exist_ok=True)
        if not os.path.isdir(model_dir):
            raise NotADirectoryError(f"Error: Could not create model directory '{model_dir}'")

        layer_paths: List[str] = []
        for i, layer in enumerate(self.layers):
            layer_name = f"layer_{i:03d}_{layer.name}"
            layer_path = layer.save(name=layer_name, path=model_dir, overwrite=overwrite)
            layer_paths.append(layer_path)

            print(f"[ModelSave] Saving {layer.name} → {layer_name}")

        model_toml_path = os.path.join(model_dir, "model.toml")
        write_toml(
            obj={
                "model": {
                    "learning_rate": self.learning_rate,
                    "loss": self.loss.name,
                    "layers": layer_paths,
                }
            },
            path=model_toml_path,
            overwrite=overwrite,
        )

        print(f"[ModelSave] Model saved → {model_toml_path}")
        return model_toml_path

    @classmethod
    def load(cls, path: str) -> Self:
        model = os.path.join(path, "model.toml")

        def __process_keys[T](obj: Dict, key: str, expected_type: Type[T]) -> T:
            data = obj.get(key, None)
            if data is None or not isinstance(data, expected_type):
                raise KeyError(f"Error: Could Not Process \"{key}\" key!")

            return data

        model_data = __process_keys(load_toml(model, find_on_path=True), "model", Dict)

        learning_rate = __process_keys(model_data, "learning_rate", float)
        loss_name = __process_keys(model_data, "loss", str)
        layer_paths = __process_keys(model_data, "layers", List)

        layers: List[ILayer] = []
        for l_path in layer_paths:
            l_type = __process_keys(load_toml(l_path), "layer_type", str)
            layers.append(Get(ILayer, label='layer', adapter=l_type, instance=False).load(l_path))

        loss = Get(ILoss, label="loss", adapter=loss_name)
        print(f"[ModelLoad] Loaded {len(layers)} layers, loss={loss_name}, lr={learning_rate}")
        return cls(layers=layers, loss=loss, learning_rate=learning_rate)

