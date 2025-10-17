from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass
class Dataset:
    x: NDArray
    y: int
