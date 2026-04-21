from simple_duck_ml.dataset_unpacker.i_dataset_unpacker import IDatasetUnpacker
from simple_duck_ml.dataset_unpacker.dataset import Dataset
from typing import Callable, Iterator, List, Literal, Optional
from numpy.typing import NDArray
from struct import unpack
import numpy as np
import struct
import cv2


class MiniBatchBinDatasetUnpacker(IDatasetUnpacker):
    """
    Memory-efficient variant of BinDatasetUnpacker that scans the binary file
    once at initialization to record the byte offset of every sample, then reads
    images from disk one mini-batch at a time via iter_batches().

    This avoids loading the entire dataset into RAM before training starts.
    The standard unpack() method is still available for small datasets or when
    all samples are needed at once.
    """

    # Fixed-size header that precedes stroke data in every sample record:
    # 8 bytes (key) + 2 bytes (country code) + 1 byte (recognized flag) + 4 bytes (timestamp)
    _HEADER_SIZE = 15

    # Canvas dimensions used when rendering sketches
    _IMAGE_SIZE = 256

    def __init__(self, path: str) -> None:
        self.path = path
        self.file_buffer = open(path, "rb")

        # Index all sample offsets upfront (single linear scan, no image data kept)
        self._offsets: List[int] = self._build_offset_index()

    # ------------------------------------------------------------------ #
    #  IDatasetUnpacker interface                                          #
    # ------------------------------------------------------------------ #

    def unpack(
        self,
        label: int,
        qnt: int | Literal["*"] = "*",
        normalization: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> Dataset:
        """
        Load a slice of samples into a single in-memory Dataset tensor.

        Loads at most qnt images (or all of them when qnt is '*').
        For large datasets prefer iter_batches() so that only one batch
        lives in RAM at a time.
        """
        if isinstance(qnt, str) and qnt != "*":
            raise ValueError("Error: 'qnt' argument should be integer or Literal['*']")

        offsets = self._select_offsets(qnt)
        images = self._decode_images_at_offsets(offsets, normalization)

        if not images:
            raise TypeError(f"Could not unpack any image from {self.file_buffer.name}!")

        # Stack individual (H, W) arrays into a single (N, H, W) tensor
        images_tensor = np.stack(images, axis=0)
        return Dataset(images_tensor, label)

    # ------------------------------------------------------------------ #
    #  Mini-batch streaming API                                            #
    # ------------------------------------------------------------------ #

    def iter_batches(
        self,
        label: int,
        batch_size: int,
        normalization: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> Iterator[Dataset]:
        """
        Yield one Dataset per mini-batch, keeping at most batch_size decoded
        images in RAM at any given time.

        The caller should consume and discard each yielded Dataset before
        requesting the next so the previous batch tensor can be garbage collected.

        Example
        -------
        for batch_dataset in unpacker.iter_batches(label=0, batch_size=32):
            model.fit([batch_dataset], epochs=1)
        """
        for batch_offsets in self._chunk_offsets(batch_size):
            images = self._decode_images_at_offsets(batch_offsets, normalization)

            if not images:
                continue

            # Only this batch is in memory; the previous batch tensor is already collectable
            batch_tensor = np.stack(images, axis=0)
            yield Dataset(batch_tensor, label)

    def read_sample(
        self,
        offset_idx: int,
        label: int,
        normalization: Optional[Callable[[NDArray], NDArray]] = None,
    ) -> Optional[Dataset]:
        """
        Decode a single sample by its position in the offset index and return
        it as a Dataset consistent with the rest of the unpacker API.

        Seeks directly to the sample without scanning the rest of the file,
        so repeated random-access calls are O(1) per seek. Used by
        Model.fit_stream() to read one image at a time during training.
        """
        offset = self._offsets[offset_idx]
        self.file_buffer.seek(offset)
        img = self._unpack_one()
        if img is None:
            return None
        if normalization is not None:
            img = normalization(img)
        return Dataset(img, label)

    def __len__(self) -> int:
        """Return the total number of samples indexed in this file."""
        return len(self._offsets)

    # ------------------------------------------------------------------ #
    #  Offset index helpers                                                #
    # ------------------------------------------------------------------ #

    def _build_offset_index(self) -> List[int]:
        """
        Perform a single linear scan of the binary file and record the byte
        position at which each sample record begins.

        Only structural bytes (header + stroke sizes) are read; pixel data is
        never allocated. The resulting list enables O(1) random access so that
        iter_batches() can seek directly to any sample without re-scanning.
        """
        offsets: List[int] = []
        self.file_buffer.seek(0)

        while True:
            sample_start = self.file_buffer.tell()

            if not self._skip_sample_bytes():
                # Reached end of file or encountered a truncated record
                break

            offsets.append(sample_start)

        return offsets

    def _skip_sample_bytes(self) -> bool:
        """
        Advance the file cursor past one complete sample record without
        decoding any image data.

        Returns True when the cursor was successfully moved past the record,
        or False when the file is exhausted or a truncated record is found.
        """
        # Skip the fixed-size header preceding stroke data
        header = self.file_buffer.read(self._HEADER_SIZE)
        if len(header) < self._HEADER_SIZE:
            return False

        n_strokes_data = self.file_buffer.read(2)
        if len(n_strokes_data) < 2:
            return False

        n_strokes, = unpack("H", n_strokes_data)

        # Each stroke: 2-byte point count + n_points bytes for x + n_points bytes for y
        for _ in range(n_strokes):
            n_points_data = self.file_buffer.read(2)
            if len(n_points_data) < 2:
                return False

            n_points, = unpack("H", n_points_data)

            x_data = self.file_buffer.read(n_points)
            y_data = self.file_buffer.read(n_points)

            if len(x_data) < n_points or len(y_data) < n_points:
                return False

        return True

    def _select_offsets(self, qnt: int | Literal["*"]) -> List[int]:
        """
        Return the slice of self._offsets that corresponds to the requested
        quantity: all offsets for '*', or the first qnt offsets otherwise.
        """
        if qnt == "*":
            return self._offsets
        return self._offsets[:qnt]

    def _chunk_offsets(self, batch_size: int) -> Iterator[List[int]]:
        """
        Yield successive non-overlapping windows of self._offsets, each of
        length batch_size. The final window may be shorter if the total number
        of samples is not divisible by batch_size.
        """
        for start in range(0, len(self._offsets), batch_size):
            yield self._offsets[start : start + batch_size]

    # ------------------------------------------------------------------ #
    #  Image decoding helpers                                              #
    # ------------------------------------------------------------------ #

    def _decode_images_at_offsets(
        self,
        offsets: List[int],
        normalization: Optional[Callable[[NDArray], NDArray]],
    ) -> List[NDArray]:
        """
        Seek to each byte offset, decode the sample at that position, and
        optionally apply normalization. Returns only the successfully decoded
        images; corrupted records are silently skipped.
        """
        images: List[NDArray] = []

        for offset in offsets:
            self.file_buffer.seek(offset)

            try:
                img = self._unpack_one()
            except struct.error:
                continue

            if img is None:
                continue

            if normalization is not None:
                img = normalization(img)

            images.append(img)

        return images

    def _unpack_one(self) -> Optional[NDArray[np.uint8]]:
        """
        Decode a single sketch from the current file cursor position by
        rendering each stroke as a sequence of connected line segments onto a
        blank white 256×256 canvas.
        """
        image = np.ones((self._IMAGE_SIZE, self._IMAGE_SIZE), dtype=np.uint8) * 255

        # Advance past the fixed header; only stroke coordinates are needed
        self.file_buffer.read(self._HEADER_SIZE)

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

            # Render each consecutive pair of points as one stroke segment
            for i in range(len(x_vec) - 1):
                pt1 = (int(x_vec[i]), int(y_vec[i]))
                pt2 = (int(x_vec[i + 1]), int(y_vec[i + 1]))
                cv2.line(image, pt1, pt2, color=(0,), thickness=3)

        return image
