from simple_duck_ml.dataset_unpacker.dataset import Dataset
from typing import Callable, Literal, Optional
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from duckdi import Interface

@Interface(label="dataset_unpacker")
class IDatasetUnpacker(ABC):
    path: str
    
    @abstractmethod
    def unpack(
        self,
        label: int,
        qnt: int | Literal["*"]="*",
        normalization: Optional[Callable[[NDArray], NDArray]]=None,
    ) -> Dataset: ...


