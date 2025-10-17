from simple_duck_ml.dataset_unpacker.i_dataset_unpacker import IDatasetUnpacker
from simple_duck_ml.dataset_unpacker.dataset import Dataset
from typing import Callable, Literal, Optional
from numpy.typing import NDArray
from struct import unpack
import numpy as np
import struct
import cv2

class BinDatasetUnpacker(IDatasetUnpacker):
    def __init__(self, path: str) -> None:
        self.path = path
        self.file_buffer = open(path, "rb")

    def unpack(
        self,
        label: int,
        qnt: int | Literal["*"]="*",
        normalization: Optional[Callable[[NDArray], NDArray]]=None,
    ) -> Dataset:
        self.file_buffer.seek(0)

        if isinstance(qnt, str) and qnt != "*":
            raise ValueError("Error: 'qnt' argument should be integer or Literal['*']")
        
        images = []
        while True:
            try:
                img = self.__unpack_one()
                if img is None:
                    break
                
                if normalization is not None:
                    img = normalization(img)
                
                images.append(img)
                if qnt != "*" and len(images) >= qnt:
                    break

            except struct.error:
                break

        if not images:
            raise TypeError(f"Could not unpack any image from {self.file_buffer.name}!")

        # Stack all images into one tensor
        images_tensor = np.stack(images, axis=0)
        return Dataset(images_tensor, label)

    def __unpack_one(self) -> Optional[NDArray[np.uint8]]:
        """
        Unpack a single sketch image from the binary dataset file.
        """
        image = np.ones((256, 256), dtype=np.uint8) * 255

        # Skip header (8 + 2 + 1 + 4 = 15 bytes)
        self.file_buffer.read(8)
        self.file_buffer.read(2)
        self.file_buffer.read(1)
        self.file_buffer.read(4)

        n_strokes_data = self.file_buffer.read(2)
        if len(n_strokes_data) < 2:
            return None

        n_strokes, = unpack("H", n_strokes_data)
        for _ in range(n_strokes):
            n_points_data = self.file_buffer.read(2)
            if len(n_points_data) < 2:
                return None

            n_points, = unpack("H", n_points_data)
            x_data = self.file_buffer.read(n_points)
            y_data = self.file_buffer.read(n_points)

            if len(x_data) < n_points or len(y_data) < n_points:
                return None

            x_vec = struct.unpack(f"{n_points}B", x_data)
            y_vec = struct.unpack(f"{n_points}B", y_data)

            for i in range(len(x_vec) - 1):
                pt1 = (int(x_vec[i]), int(y_vec[i]))
                pt2 = (int(x_vec[i + 1]), int(y_vec[i + 1]))
                cv2.line(image, pt1, pt2, color=(0,), thickness=3)

        return image

