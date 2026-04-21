from simple_duck_ml.dataset_unpacker.i_data_source import IDataSource
from simple_duck_ml.dataset_unpacker.dataset import Dataset
from typing import Optional


class InMemoryDataSource(IDataSource):
    """Wraps a pre-loaded Dataset for use with Model.fit()."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset.x)

    def get_sample(self, idx: int) -> Optional[Dataset]:
        return Dataset(self._dataset.x[idx], self._dataset.y)
