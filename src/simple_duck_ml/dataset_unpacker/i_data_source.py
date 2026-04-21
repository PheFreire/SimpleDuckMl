from simple_duck_ml.dataset_unpacker.dataset import Dataset
from abc import ABC, abstractmethod
from typing import Optional


class IDataSource(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_sample(self, idx: int) -> Optional[Dataset]: ...
