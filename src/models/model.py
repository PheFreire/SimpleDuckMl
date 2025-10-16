from datasets.dataset import Dataset
from layers.i_layer import ILayer
from numpy.typing import NDArray
from loss.i_loss import ILoss
from tqdm import trange
from typing import List
import numpy as np


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

