"""Stateless, sequence-consistent spatial and photometric augmentations."""

import hashlib
import math
from dataclasses import dataclass, replace

import numpy as np

from actin_tracking.data.geometry import GeometryOp, GeometryTrace, Size
from actin_tracking.data.samples import FrameSample, transform_frame


@dataclass(frozen=True)
class Augmentations:
    seed: int = 20261117
    resize: Size | None = None
    pad_to: Size | None = None
    crop_size: Size | None = None
    horizontal_flip_probability: float = 0.0
    vertical_flip_probability: float = 0.0
    brightness: tuple[float, float] = (1.0, 1.0)
    contrast: tuple[float, float] = (1.0, 1.0)
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        for size in (self.resize, self.pad_to, self.crop_size):
            if size is not None and (
                len(size) != 2 or any(type(v) is not int or v <= 0 for v in size)
            ):
                raise ValueError("Sizes must be positive (height, width) integer pairs")
        for value in (self.horizontal_flip_probability, self.vertical_flip_probability):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Flip probability must lie in [0,1]")
        for interval in (self.brightness, self.contrast):
            if not all(math.isfinite(v) for v in interval) or not 0 <= interval[0] <= interval[1]:
                raise ValueError("Photometric ranges must be finite, nonnegative and ordered")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")

    def apply(self, frame: FrameSample) -> FrameSample:
        key = frame.metadata.sequence_id or frame.metadata.file_name
        seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{frame.metadata.split}:{key}".encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(seed)
        ops: list[GeometryOp] = []
        trace = GeometryTrace(frame.geometry.size)
        if self.resize is not None:
            ops.append(GeometryOp("resize", self.resize))
            trace = trace.append((ops[-1],))
        if self.pad_to is not None:
            height, width = trace.size
            if self.pad_to[0] < height or self.pad_to[1] < width:
                raise ValueError("pad_to cannot shrink an image")
            ops.append(GeometryOp("pad", (0, 0, self.pad_to[1] - width, self.pad_to[0] - height)))
            trace = trace.append((ops[-1],))
        if self.crop_size is not None:
            height, width = trace.size
            crop_height, crop_width = self.crop_size
            if crop_height > height or crop_width > width:
                raise ValueError("Crop exceeds current image size")
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            ops.append(GeometryOp("crop", (left, top, crop_width, crop_height)))
        if rng.random() < self.horizontal_flip_probability:
            ops.append(GeometryOp("hflip"))
        if rng.random() < self.vertical_flip_probability:
            ops.append(GeometryOp("vflip"))
        result = transform_frame(frame, tuple(ops)) if ops else frame
        brightness = float(rng.uniform(*self.brightness))
        contrast = float(rng.uniform(*self.contrast))
        noise_seed = int(rng.integers(0, 2**32))
        if brightness != 1 or contrast != 1 or self.noise_std:
            values = (result.image.astype(np.float32) - 127.5) * contrast + 127.5
            values *= brightness
            if self.noise_std:
                values += (
                    np.random.default_rng(noise_seed)
                    .normal(0, self.noise_std, values.shape)
                    .astype(np.float32)
                )
            image = result.image.copy()
            image[result.valid_pixels] = np.clip(
                np.rint(values[result.valid_pixels]), 0, 255
            ).astype(np.uint8)
            result = replace(result, image=image)
        return replace(
            result,
            photometric_history=result.photometric_history
            + (
                {
                    "brightness": brightness,
                    "contrast": contrast,
                    "noise_std": self.noise_std,
                    "noise_seed": noise_seed,
                },
            ),
        )
