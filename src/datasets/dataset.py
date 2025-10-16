from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

@dataclass
class Dataset:
    x: NDArray[np.float64]
    y: int
