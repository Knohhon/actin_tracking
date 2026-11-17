"""Sampled descriptive diagnostics, never benchmark length/crossing ground truth."""

import hashlib
import json
import sqlite3
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask  # type: ignore[import-untyped]


def skeleton_pixels(mask: Any) -> int:
    """Zhang-Suen thinning; count centreline pixels (no subpixel length claim)."""
    image = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    while True:
        removed = 0
        for phase in (0, 1):
            p = [
                image[:-2, 1:-1],
                image[:-2, 2:],
                image[1:-1, 2:],
                image[2:, 2:],
                image[2:, 1:-1],
                image[2:, :-2],
                image[1:-1, :-2],
                image[:-2, :-2],
            ]
            neighbours = sum(p)
            transitions = sum((a == 0) & (b == 1) for a, b in zip(p, p[1:] + p[:1], strict=False))
            if phase == 0:
                constraint = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                constraint = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            drop = (image[1:-1, 1:-1] == 1) & (neighbours >= 2) & (neighbours <= 6)
            drop &= (transitions == 1) & constraint
            removed += int(drop.sum())
            image[1:-1, 1:-1][drop] = 0
        if removed == 0:
            return int(image.sum())


def overlap_sample(db: sqlite3.Connection, limit: int = 3) -> dict[str, Any]:
    """Count positive-area mask-overlap pairs on up to three physical frames."""
    frames = []
    db.create_function("frame_hash", 1, lambda value: hashlib.sha256(value.encode()).hexdigest())
    for filename, identifier in db.execute(
        "SELECT file_name,canonical_image_id FROM physical_frames WHERE valid=1 "
        "AND conflicted=0 AND sha256 IS NOT NULL ORDER BY frame_hash(file_name) LIMIT ?",
        (limit,),
    ):
        rles = []
        for (payload,) in db.execute(
            "SELECT payload FROM annotations WHERE image_id=? AND mask_hash IS NOT NULL "
            "ORDER BY row_id LIMIT 2049",
            (identifier,),
        ):
            rle = json.loads(payload)["segmentation"]
            if isinstance(rle["counts"], list):
                rle = coco_mask.frPyObjects(rle, *rle["size"])
            rles.append(rle)
        if len(rles) > 2048:
            frames.append({"file_name": filename, "skipped": "more than 2048 instances"})
            continue
        overlaps = coco_mask.iou(rles, rles, [0] * len(rles)) if rles else np.zeros((0, 0))
        frames.append(
            {
                "file_name": filename,
                "instances": len(rles),
                "overlap_pairs": int(np.triu(np.asarray(overlaps) > 0, k=1).sum()),
            }
        )
    return {
        "frames": frames,
        "selection": "lowest SHA256 filename, unique physical frames",
        "meaning": "Positive mask IoU pairs; overlaps are not necessarily centreline crossings.",
    }
