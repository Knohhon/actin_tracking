"""Image-edge XY geometry and explicitly lossy raster transformations."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

Size = tuple[int, int]  # height, width


@dataclass(frozen=True)
class GeometryOp:
    kind: Literal["resize", "pad", "crop", "hflip", "vflip"]
    values: tuple[int, ...] = ()


@dataclass(frozen=True)
class GeometryStep:
    op: GeometryOp
    before: Size
    after: Size

    def matrix(self) -> NDArray[np.float64]:
        height, width = self.before
        op = self.op
        matrix = np.eye(3, dtype=np.float64)
        if op.kind == "resize":
            matrix[0, 0], matrix[1, 1] = self.after[1] / width, self.after[0] / height
        elif op.kind == "pad":
            matrix[:2, 2] = op.values[:2]
        elif op.kind == "crop":
            matrix[:2, 2] = -np.asarray(op.values[:2])
        elif op.kind == "hflip":
            matrix[0, 0], matrix[0, 2] = -1, width
        elif op.kind == "vflip":
            matrix[1, 1], matrix[1, 2] = -1, height
        return matrix


def make_step(size: Size, op: GeometryOp) -> GeometryStep:
    height, width = size
    values = op.values
    if any(type(value) is not int for value in values):
        raise ValueError("Geometry parameters must be integers")
    if op.kind == "resize" and len(values) == 2 and min(values) > 0:
        after = (values[0], values[1])
    elif op.kind == "pad" and len(values) == 4 and min(values) >= 0:
        left, top, right, bottom = values
        after = (height + top + bottom, width + left + right)
    elif op.kind == "crop" and len(values) == 4:
        left, top, crop_width, crop_height = values
        if (
            min(left, top) < 0
            or min(crop_width, crop_height) <= 0
            or left + crop_width > width
            or top + crop_height > height
        ):
            raise ValueError("Crop must lie within the current image")
        after = (crop_height, crop_width)
    elif op.kind in ("hflip", "vflip") and not values:
        after = size
    else:
        raise ValueError(f"Invalid geometry operation: {op}")
    if after[0] * after[1] > 64_000_000:
        raise ValueError("Transformed image exceeds 64 million pixels")
    return GeometryStep(op, size, after)


def apply_step(array: NDArray[Any], step: GeometryStep, *, mask: bool) -> NDArray[Any]:
    op = step.op
    if tuple(array.shape) != step.before:
        raise ValueError("Array dimensions do not match geometry history")
    if op.kind == "resize":
        method = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
        return np.asarray(Image.fromarray(array).resize(step.after[::-1], method)).copy()
    if op.kind == "pad":
        left, top, right, bottom = op.values
        return np.pad(array, ((top, bottom), (left, right)), constant_values=0)
    if op.kind == "crop":
        left, top, width, height = op.values
        return array[top : top + height, left : left + width].copy()
    if op.kind == "hflip":
        return array[:, ::-1].copy()
    return array[::-1, :].copy()


@dataclass(frozen=True)
class GeometryTrace:
    original_size: Size
    steps: tuple[GeometryStep, ...] = ()

    def __post_init__(self) -> None:
        if (
            len(self.original_size) != 2
            or any(type(v) is not int or v <= 0 for v in self.original_size)
            or self.original_size[0] * self.original_size[1] > 64_000_000
        ):
            raise ValueError("Invalid original image dimensions")
        size = self.original_size
        for step in self.steps:
            if step.before != size or make_step(size, step.op) != step:
                raise ValueError("Inconsistent geometry history")
            size = step.after

    @property
    def size(self) -> Size:
        return self.steps[-1].after if self.steps else self.original_size

    @property
    def matrix(self) -> NDArray[np.float64]:
        matrix = np.eye(3, dtype=np.float64)
        for step in self.steps:
            matrix = step.matrix() @ matrix
        return matrix

    def append(self, ops: tuple[GeometryOp, ...]) -> "GeometryTrace":
        steps = list(self.steps)
        size = self.size
        for op in ops:
            step = make_step(size, op)
            steps.append(step)
            size = step.after
        return GeometryTrace(self.original_size, tuple(steps))

    def points(self, points: NDArray[Any], *, inverse: bool = False) -> NDArray[np.float64]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("Expected finite Nx2 XY coordinates")
        matrix = np.linalg.inv(self.matrix) if inverse else self.matrix
        result = np.column_stack((values, np.ones(len(values)))) @ matrix.T
        return np.asarray(result[:, :2], dtype=np.float64)

    def boxes(
        self, boxes: NDArray[Any], *, inverse: bool = False, clip: bool = False
    ) -> NDArray[np.float64]:
        values = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if not np.isfinite(values).all() or (values[:, 2:] < 0).any():
            raise ValueError("Expected finite nonnegative box extents")
        x, y, w, h = values.T
        corners = np.stack(
            (
                np.stack((x, y), 1),
                np.stack((x + w, y), 1),
                np.stack((x, y + h), 1),
                np.stack((x + w, y + h), 1),
            ),
            1,
        )
        mapped = self.points(corners.reshape(-1, 2), inverse=inverse).reshape(-1, 4, 2)
        lower, upper = mapped.min(1), mapped.max(1)
        if clip:
            height, width = self.original_size if inverse else self.size
            lower = np.clip(lower, (0, 0), (width, height))
            upper = np.clip(upper, (0, 0), (width, height))
        return np.column_stack((lower, upper - lower))

    def restore_mask(self, mask: NDArray[Any]) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        """Return original-grid mask and visibility; discarded crop pixels remain unknown."""
        values = np.asarray(mask, dtype=np.uint8)
        if tuple(values.shape) != self.size:
            raise ValueError("Mask dimensions do not match transformed image")
        visible = np.ones_like(values, dtype=np.uint8)
        for step in reversed(self.steps):
            if step.op.kind == "resize":
                reverse = make_step(step.after, GeometryOp("resize", step.before))
            elif step.op.kind == "pad":
                left, top, _, _ = step.op.values
                reverse = make_step(
                    step.after, GeometryOp("crop", (left, top, step.before[1], step.before[0]))
                )
            elif step.op.kind == "crop":
                left, top, width, height = step.op.values
                reverse = make_step(
                    step.after,
                    GeometryOp(
                        "pad",
                        (left, top, step.before[1] - left - width, step.before[0] - top - height),
                    ),
                )
            else:
                reverse = make_step(step.after, step.op)
            values = apply_step(values, reverse, mask=True)
            visible = apply_step(visible, reverse, mask=True)
        return values, visible.astype(bool)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_size": list(self.original_size),
            "size": list(self.size),
            "matrix_xy": self.matrix.tolist(),
            "steps": [
                {
                    "kind": s.op.kind,
                    "values": list(s.op.values),
                    "before": list(s.before),
                    "after": list(s.after),
                }
                for s in self.steps
            ],
        }
