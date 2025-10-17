from typing import Dict
import toml
import os

def write_toml(obj: Dict, path: str, overwrite: bool=True) -> str:
    path = path if path.endswith(".toml") else f"{path}.toml"
    if os.path.isfile(path) and not overwrite:
        raise FileExistsError(f"Error: \"{path}\" already exists\nSet \"overwrite=True\" to overwrite file!")

    with open(path, "w") as file:
        file.write(toml.dumps(obj))

    return path


def load_toml(path: str, find_on_path: bool=False) -> Dict:
    if find_on_path and not path.endswith(".toml") and os.path.isdir(path):
        files = [file for file in os.listdir(path) if file.endswith(".toml")]
        path = files[0] if files else path

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Error: Could not find \".toml\" file on \"{path}\"!")   

    with open(path, "r") as file:
        data = toml.load(file)

    return data
