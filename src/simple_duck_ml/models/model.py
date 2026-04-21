import uuid
from simple_duck_ml.serializers.create_dir import create_dir
from simple_duck_ml.serializers.toml_io import load_toml, write_toml
from simple_duck_ml.dataset_unpacker.i_data_source import IDataSource
from typing import Dict, List, Optional, Self, Sequence, Type
from simple_duck_ml.layers.i_layer import ILayer
from simple_duck_ml.loss.i_loss import ILoss
from numpy.typing import NDArray
from tqdm import trange
from duckdi import Get
import numpy as np
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

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred: NDArray[np.float32], y_true: NDArray[np.float32]) -> None:
        delta = self.loss.derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self, batch_size: int = 1) -> None:
        for layer in self.layers:
            layer.update(self.learning_rate, batch_size)

    def _one_hot(self, label: int, num_classes: int) -> NDArray[np.float32]:
        one_hot = np.zeros((num_classes, 1), dtype=np.float32)
        one_hot[label, 0] = 1.0
        return one_hot

    def fit(
        self,
        sources: Sequence[IDataSource],
        epochs: int = 5,
        batch_size: int = 10,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> None:
        # Cada source representa uma classe de treinamento
        # O indice da source é o valor numerico referente ao label
        num_classes = len(sources)

        # Monta um indice global leve: (source_idx, sample_idx)
        # Apenas dois inteiros por amostra (nenhuma imagem é carregada aqui)
        all_indices = np.array(
            [
                (s_idx, i) for s_idx, source in enumerate(sources) 
                for i in range(len(source))
            ],
            dtype=np.int64,
        )
        total_samples = len(all_indices)

        for epoch in trange(epochs, desc="Epoch"):
            # Embaralha os indices para que cada epoch veja as amostras em ordem diferente,
            # evitando que o modelo aprenda a ordem dos dados ao invés dos padrões
            if shuffle:
                np.random.shuffle(all_indices)

            total_loss = 0.0

            # Divide os indices em fatias de tamanho batch_size
            for batch_start in range(0, total_samples, batch_size):
                batch = all_indices[batch_start:batch_start + batch_size]
                batch_loss = 0.0
                actual_batch_size = 0  # conta amostras validas (descarta leituras corrompidas)

                for s_idx, i in batch:
                    # Carrega a imagem do disco ou memoria conforme o tipo de IDataSource
                    sample = sources[s_idx].get_sample(int(i))
                    if sample is None:
                        continue

                    # Converte o label inteiro para vetor one-hot: ex. classe 2 de 3 → [0, 0, 1]
                    y_true = self._one_hot(sample.y, num_classes)

                    # Forward: propaga a imagem por todas as layers e retorna as probabilidades
                    y_pred = self.forward(sample.x)

                    # Calcula o erro entre a predição e o valor esperado
                    loss = self.loss(y_pred, y_true)
                    batch_loss += float(loss)

                    # Backward: calcula o gradiente do erro em relação a cada peso da rede
                    # Os gradientes são acumulados em cada layer (∇W += ...) sem atualizar ainda
                    self.backward(y_pred, y_true)
                    actual_batch_size += 1

                # Atualiza os pesos uma única vez com a média dos gradientes acumulados no batch
                # W -= learning_rate * (∇W / actual_batch_size)
                if actual_batch_size > 0:
                    self.update(actual_batch_size)

                total_loss += batch_loss

            avg_loss = total_loss / total_samples
            if verbose:
                print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")
                print("-=" * 30)

    def save(self, name: Optional[str] = None, path: str = ".", overwrite: bool = True) -> str:
        name = str(uuid.uuid4()).replace("-", "") if name is None else name
        dir = create_dir(name, path, overwrite)

        layer_paths: List[str] = []
        for i, layer in enumerate(self.layers):
            layer_name = f"layer_{i:03d}_{layer.name}"
            layer_path = layer.save(name=layer_name, path=dir, overwrite=overwrite)
            layer_paths.append(layer_path['relative'])

            print(f"[ModelSave] Saving {layer.name} -> {layer_name}")

        model_toml_path = os.path.join(dir, "model.toml")
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

        model_data = __process_keys(load_toml(model), "model", Dict)

        learning_rate = __process_keys(model_data, "learning_rate", float)
        loss_name = __process_keys(model_data, "loss", str)
        layer_paths = __process_keys(model_data, "layers", List)

        layers: List[ILayer] = []
        for raw_l_path in layer_paths:
            l_path = os.path.join(path, raw_l_path)
            l_type = __process_keys(load_toml(l_path), "layer_type", str)
            layers.append(Get(ILayer, label='layer', adapter=l_type, instance=False).load(l_path))

        loss = Get(ILoss, label="loss", adapter=loss_name)
        print(f"[ModelLoad] Loaded {len(layers)} layers, loss={loss_name}, lr={learning_rate}")
        return cls(layers=layers, loss=loss, learning_rate=learning_rate)

