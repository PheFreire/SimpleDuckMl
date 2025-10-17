from numpy.typing import NDArray
from typing import Dict
import numpy as np
import os

def write_tensor(tensors: Dict[str, NDArray[np.float64]], path: str, overwrite: bool=True) -> str:
    path = path if path.endswith(".npz") else f"{path}.npz"
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(f"Error: \"{path}\" already exists\nSet \"overwrite=True\" to overwrite file!")

    np.savez_compressed(file=path, allow_pickle=True, **tensors)
    return path

    
def load_tensor(path: str, find_on_path: bool) -> Dict[str, NDArray[np.float64]]:
    if find_on_path and not path.endswith(".npz") and os.path.isdir(path):
        files = [file for file in os.listdir(path) if file.endswith(".npz")]
        if files:
           path = os.path.join(path, files[0])

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Error: Could not find \".npz\" file on \"{path}\"!")   

    with np.load(path, allow_pickle=True) as data:
        tensors = { key: data[key] for key in data.files }

    return tensors   
