"""Loaded frames with exact identifiers, lazy masks, and reversible coordinates."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as coco_mask  # type: ignore[import-untyped]

from actin_tracking.data.geometry import GeometryOp, GeometryTrace, apply_step
from actin_tracking.data.rle import compressed_rle, decode_rle, encode_rle
from actin_tracking.data.schemas import MaskRLE

Box = tuple[float, float, float, float]


class LazyMask:
    """Retain source RLE and geometry; never keep a collection of decoded masks."""

    def __init__(self, rle: dict[str, Any], geometry: GeometryTrace | None = None) -> None:
        self._rle = deepcopy(rle)
        if isinstance(self._rle.get("counts"), bytes):
            self._rle["counts"] = self._rle["counts"].decode("ascii")
        if not isinstance(rle.get("size"), list) or len(rle["size"]) != 2:
            raise ValueError("Mask requires RLE dimensions")
        self.geometry = geometry or GeometryTrace(tuple(rle["size"]))
        if tuple(rle["size"]) != self.geometry.original_size:
            raise ValueError("Mask and original image dimensions disagree")

    @property
    def size(self) -> tuple[int, int]:
        return self.geometry.size

    @property
    def original_rle(self) -> dict[str, Any]:
        return deepcopy(self._rle)

    def decode(self) -> NDArray[np.uint8]:
        mask = decode_rle(self._rle)
        for step in self.geometry.steps:
            mask = apply_step(mask, step, mask=True)
        return mask

    def to_rle(self) -> MaskRLE:
        if not self.geometry.steps:
            return compressed_rle(self._rle)
        return encode_rle(self.decode())

    def transformed(self, ops: tuple[GeometryOp, ...]) -> "LazyMask":
        return LazyMask(self._rle, self.geometry.append(ops))

    def bbox_xywh(self) -> Box:
        box = coco_mask.toBbox(self.to_rle()).tolist()
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


@dataclass(frozen=True)
class InstanceSample:
    instance_id: str
    category_id: str
    track_id: str | None
    bbox_xywh: Box
    original_bbox_xywh: Box
    mask: LazyMask
    iscrowd: bool = False


@dataclass(frozen=True)
class FrameMetadata:
    split: str
    source: str
    image_id: str
    file_name: str
    source_image_ids: tuple[str, ...]
    sequence_id: str | None
    frame_index: int
    temporal_continuity_confirmed: bool
    window_id: str | None = None
    fps: float | None = None
    pixel_size_um: float | None = None
    calibration_provenance: str | None = None


@dataclass(frozen=True)
class FrameSample:
    image: NDArray[np.uint8]
    valid_pixels: NDArray[np.bool_]
    instances: tuple[InstanceSample, ...]
    metadata: FrameMetadata
    geometry: GeometryTrace
    photometric_history: tuple[dict[str, Any], ...] = ()
    dropped_instance_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.image.ndim != 2 or self.image.dtype != np.uint8:
            raise ValueError("Expected a uint8 grayscale image")
        if self.valid_pixels.dtype != bool or self.valid_pixels.shape != self.image.shape:
            raise ValueError("valid_pixels must be a boolean array matching image dimensions")
        if tuple(self.image.shape) != self.geometry.size:
            raise ValueError("Image and geometry dimensions disagree")


def transform_frame(frame: FrameSample, ops: tuple[GeometryOp, ...]) -> FrameSample:
    """Apply identical geometry to image, valid pixels, boxes and lazy instance masks."""
    local = GeometryTrace(frame.geometry.size).append(ops)
    image, valid = frame.image, frame.valid_pixels.astype(np.uint8)
    for step in local.steps:
        image = apply_step(image, step, mask=False)
        valid = apply_step(valid, step, mask=True)
    instances = []
    dropped = list(frame.dropped_instance_ids)
    for instance in frame.instances:
        mask = instance.mask.transformed(ops)
        # Lossy operations can remove an entire thin instance; decode only for visibility,
        # one mask at a time, without caching the resulting arrays.
        if any(op.kind in ("crop", "resize") for op in ops) and not mask.decode().any():
            dropped.append(instance.instance_id)
            continue
        box = local.boxes(np.asarray([instance.bbox_xywh]), clip=True)[0]
        instances.append(
            replace(
                instance,
                mask=mask,
                bbox_xywh=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )
    return replace(
        frame,
        image=image.copy(),
        valid_pixels=valid.astype(bool),
        instances=tuple(instances),
        geometry=frame.geometry.append(ops),
        dropped_instance_ids=tuple(dropped),
    )


@dataclass(frozen=True)
class FrameBatch:
    images: NDArray[np.uint8]  # N,H,W; conversion/normalization belongs to model adapters
    valid_pixels: NDArray[np.bool_]
    frames: tuple[FrameSample, ...]
    sizes_before_batch: tuple[tuple[int, int], ...]


def collate_frames(frames: list[FrameSample], *, allow_duplicates: bool = False) -> FrameBatch:
    if not frames:
        raise ValueError("Cannot collate an empty batch")
    keys = [(f.metadata.split, f.metadata.file_name) for f in frames]
    if not allow_duplicates and len(set(keys)) != len(keys):
        raise ValueError(
            "Duplicate physical frames in batch; overlapping windows are not independent"
        )
    sizes = tuple(f.geometry.size for f in frames)
    height, width = max(s[0] for s in sizes), max(s[1] for s in sizes)
    padded = tuple(
        transform_frame(
            f, (GeometryOp("pad", (0, 0, width - f.geometry.size[1], height - f.geometry.size[0])),)
        )
        for f in frames
    )
    return FrameBatch(
        np.stack([f.image for f in padded]),
        np.stack([f.valid_pixels for f in padded]),
        padded,
        sizes,
    )


@dataclass(frozen=True)
class SequenceBatch:
    images: NDArray[np.uint8]  # B,T,H,W
    valid_pixels: NDArray[np.bool_]
    valid_frames: NDArray[np.bool_]  # False for padded temporal slots
    sequences: tuple[tuple[FrameSample, ...], ...]


def collate_sequences(
    sequences: list[list[FrameSample]], *, allow_duplicates: bool = False
) -> SequenceBatch:
    """Pad time/spatial axes of genuine sequences, preserving IDs and every frame index."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Cannot collate empty temporal sequences")
    flat = [frame for sequence in sequences for frame in sequence]
    keys = [(f.metadata.split, f.metadata.file_name) for f in flat]
    if not allow_duplicates and len(set(keys)) != len(keys):
        raise ValueError("Duplicate physical frames in temporal batch")
    for sequence in sequences:
        first = sequence[0]
        if first.metadata.sequence_id is None or not first.metadata.temporal_continuity_confirmed:
            raise ValueError("Temporal batches require confirmed sequences, not real windows")
        for position, frame in enumerate(sequence):
            if (
                frame.metadata.sequence_id != first.metadata.sequence_id
                or frame.metadata.split != first.metadata.split
                or not frame.metadata.temporal_continuity_confirmed
            ):
                raise ValueError("A temporal sample must belong to one confirmed sequence")
            if frame.geometry != first.geometry:
                raise ValueError("Temporal frames must share spatial transformations")
            if (
                position
                and frame.metadata.frame_index != sequence[position - 1].metadata.frame_index + 1
            ):
                raise ValueError(
                    "Temporal frames must be consecutive and ordered; gaps need explicit handling"
                )
    height, width = max(f.geometry.size[0] for f in flat), max(f.geometry.size[1] for f in flat)
    if height * width > 64_000_000:
        raise ValueError("Temporal spatial padding exceeds the image pixel limit")
    shape = (len(sequences), max(len(s) for s in sequences), height, width)
    images = np.zeros(shape, dtype=np.uint8)
    valid_pixels = np.zeros(shape, dtype=bool)
    valid_frames = np.zeros(shape[:2], dtype=bool)
    padded_sequences = []
    for b, sequence in enumerate(sequences):
        padded = []
        for t, frame in enumerate(sequence):
            current_height, current_width = frame.geometry.size
            result = transform_frame(
                frame, (GeometryOp("pad", (0, 0, width - current_width, height - current_height)),)
            )
            images[b, t], valid_pixels[b, t], valid_frames[b, t] = (
                result.image,
                result.valid_pixels,
                True,
            )
            padded.append(replace(result, image=images[b, t], valid_pixels=valid_pixels[b, t]))
        padded_sequences.append(tuple(padded))
    return SequenceBatch(images, valid_pixels, valid_frames, tuple(padded_sequences))
