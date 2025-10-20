from simple_duck_ml.layers.i_layer import ILayer
from abc import ABC, abstractmethod
from typing import Dict, Optional


class ILayerWriter(ABC):
    @abstractmethod
    def __call__(self, layer: ILayer, name: Optional[str], path: str, overwrite: bool) -> Dict[str, str]: ...
