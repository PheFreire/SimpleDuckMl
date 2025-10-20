from simple_duck_ml.serializers.create_dir import create_dir
from simple_duck_ml.serializers.tensor_io import write_tensor
from simple_duck_ml.serializers.toml_io import write_toml
from simple_duck_ml.io.i_layer_writer import ILayerWriter
from simple_duck_ml.layers.i_layer import ILayer
from typing import Dict, Optional
from uuid import uuid4
import os


class TomlLayerWriter(ILayerWriter):
    def __call__(self, layer: ILayer, name: Optional[str], path: str, overwrite: bool) -> Dict[str, str]:
        name = str(uuid4()).replace("-", "") if name is None else name

        dir_path = create_dir(name, path, overwrite)
        absolute_file_path = os.path.join(dir_path, name)
        to_write = {}

        tensors = layer.info.get("tensors", None)
        if tensors is not None:
            absolute_tensor_path = write_tensor(
                tensors=tensors,
                path=absolute_file_path,
                overwrite=overwrite
            )
            to_write["tensors_path"] = { 
                "absolute": absolute_tensor_path,
                "relative": os.path.join(name, f"{name}.npz")
            }

        metadata = layer.info.get("metadata", None)
        if metadata is not None:
            to_write = { **metadata, **to_write }

        absolute_path = write_toml(
            obj=to_write,
            path=absolute_file_path,
            overwrite=overwrite,
        )
        
        return { "absolute": absolute_path, "relative": os.path.join(name, f"{name}.toml") }


