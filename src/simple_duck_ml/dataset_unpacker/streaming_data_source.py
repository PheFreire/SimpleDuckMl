from simple_duck_ml.dataset_unpacker.i_data_source import IDataSource
from simple_duck_ml.dataset_unpacker.minibatch_bin_dataset_unpacker import MiniBatchBinDatasetUnpacker
from simple_duck_ml.dataset_unpacker.dataset import Dataset
from typing import Callable, Optional
from numpy.typing import NDArray


class StreamingDataSource(IDataSource):
    """Reads samples from disk on demand via MiniBatchBinDatasetUnpacker."""

    def __init__(
        self,
        unpacker: MiniBatchBinDatasetUnpacker,
        label: int,
        normalization: Optional[Callable[[NDArray], NDArray]] = None,
        limit: Optional[int] = None,
    ) -> None:
        self._unpacker = unpacker
        self._label = label
        self._normalization = normalization
        self._limit = limit

    def __len__(self) -> int:
        total = len(self._unpacker)
        return min(total, self._limit) if self._limit is not None else total

    def get_sample(self, idx: int) -> Optional[Dataset]:
        return self._unpacker.read_sample(idx, self._label, self._normalization)
