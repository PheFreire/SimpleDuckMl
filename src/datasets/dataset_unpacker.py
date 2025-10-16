from typing import Literal, Optional
from datasets.dataset import Dataset
from numpy.typing import NDArray
from struct import unpack
import numpy as np
import struct
import cv2


class DatasetUnpacker:
    def __init__(self, filename: str) -> None:
        self.file_handle = open(filename, "rb")
        self.file_handle.seek(0)

    def unpack(
        self,
        label: int,
        qnt: int | Literal["*"] = "*",
        image_size: int = 64
    ) -> Dataset:
        if isinstance(qnt, str) and qnt != "*":
            raise ValueError("Error: 'qnt' argument should be integer or Literal['*']")

        images: list[NDArray[np.float64]] = []
        while True:
            try:
                res = self.unpack_one()
                if res is None:
                    break

                # Resize | normalize | invert to black background
                img = cv2.resize(res, (image_size, image_size)).astype(np.float64)
                img = 1.0 - (img / 255.0)
                images.append(img)

                if qnt != "*" and len(images) >= qnt:
                    break

            except struct.error:
                break

        if not images:
            raise TypeError(f"Could not unpack any image from {self.file_handle.name}!")

        # Stack all images into one tensor
        images_tensor = np.stack(images, axis=0)
        return Dataset(images_tensor, label)

    def unpack_one(self, base_size: int = 256) -> Optional[NDArray[np.uint8]]:
        """
        Unpack a single sketch image from the binary dataset file.
        """
        image = np.ones((base_size, base_size), dtype=np.uint8) * 255

        # Skip header (8 + 2 + 1 + 4 = 15 bytes)
        self.file_handle.read(8)
        self.file_handle.read(2)
        self.file_handle.read(1)
        self.file_handle.read(4)

        n_strokes_data = self.file_handle.read(2)
        if len(n_strokes_data) < 2:
            return None

        n_strokes, = unpack("H", n_strokes_data)
        for _ in range(n_strokes):
            n_points_data = self.file_handle.read(2)
            if len(n_points_data) < 2:
                return None

            n_points, = unpack("H", n_points_data)
            x_data = self.file_handle.read(n_points)
            y_data = self.file_handle.read(n_points)

            if len(x_data) < n_points or len(y_data) < n_points:
                return None

            x_vec = struct.unpack(f"{n_points}B", x_data)
            y_vec = struct.unpack(f"{n_points}B", y_data)

            for i in range(len(x_vec) - 1):
                pt1 = (int(x_vec[i]), int(y_vec[i]))
                pt2 = (int(x_vec[i + 1]), int(y_vec[i + 1]))
                cv2.line(image, pt1, pt2, color=(0,), thickness=3)

        return image

    def __del__(self):
        if hasattr(self, "file_handle") and not self.file_handle.closed:
            self.file_handle.close()

