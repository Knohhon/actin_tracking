"""Safe COCO RLE round trips without coercing identifiers or geometry."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as coco_mask  # type: ignore[import-untyped]

from actin_tracking.data.schemas import MaskRLE


def validate_runs(rle: dict[str, Any]) -> int:
    """Validate compressed run totals before invoking the native COCO decoder."""
    counts = rle["counts"]
    if isinstance(counts, str):
        runs: list[int] = []
        offset = 0
        while offset < len(counts):
            value = 0
            shift = 0
            while True:
                if offset >= len(counts) or shift > 63:
                    raise ValueError("Truncated or oversized RLE run")
                char = ord(counts[offset]) - 48
                offset += 1
                if not 0 <= char <= 63:
                    raise ValueError("Invalid compressed RLE character")
                value |= (char & 31) << shift
                shift += 5
                if not char & 32:
                    if char & 16:
                        value |= -1 << shift
                    break
            if len(runs) > 2:
                value += runs[-2]
            if value < 0:
                raise ValueError("Negative decoded RLE run")
            runs.append(value)
    elif isinstance(counts, list) and all(type(v) is int and v >= 0 for v in counts):
        runs = counts
    else:
        raise ValueError("Invalid RLE counts")
    size = rle["size"]
    if sum(runs) != size[0] * size[1]:
        raise ValueError("RLE runs do not cover the declared image size")
    return sum(runs[1::2])


def validate_rle(rle: dict[str, Any]) -> None:
    size = rle.get("size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
    ):
        raise ValueError("RLE size must be [positive height, positive width]")
    if size[0] * size[1] > 64_000_000:
        raise ValueError("RLE exceeds 64 million pixel decode limit")
    validate_runs(rle)


def compressed_rle(rle: dict[str, Any]) -> MaskRLE:
    if isinstance(rle.get("counts"), bytes):
        rle = {**rle, "counts": rle["counts"].decode("ascii")}
    validate_rle(rle)
    if isinstance(rle["counts"], list):
        encoded = coco_mask.frPyObjects(rle, *rle["size"])
        counts = encoded["counts"].decode("ascii")
    else:
        counts = rle["counts"]
    return {"size": list(rle["size"]), "counts": counts}


def decode_rle(rle: dict[str, Any]) -> NDArray[np.uint8]:
    result = np.asarray(coco_mask.decode(compressed_rle(rle)), dtype=np.uint8)
    if result.ndim != 2:
        raise ValueError("Expected a single instance mask")
    return result


def encode_rle(mask: NDArray[Any]) -> MaskRLE:
    values = np.asarray(mask)
    if values.ndim != 2 or 0 in values.shape or not np.all((values == 0) | (values == 1)):
        raise ValueError("Expected a nonempty 2D binary mask")
    encoded = coco_mask.encode(np.asfortranarray(values, dtype=np.uint8))
    return {"size": list(values.shape), "counts": encoded["counts"].decode("ascii")}
