"""Streaming grayscale Dataset API over an audited, read-only SQLite cache."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from actin_tracking.data.augmentations import Augmentations
from actin_tracking.data.geometry import GeometryTrace
from actin_tracking.data.indexing import DatasetIndex
from actin_tracking.data.samples import FrameMetadata, FrameSample, InstanceSample, LazyMask
from actin_tracking.data.schemas import exact_id


class FilamentDataset:
    def __init__(
        self, index_path: Path, images_root: Path, augmentations: Augmentations | None = None
    ) -> None:
        self.index = DatasetIndex(index_path)
        self.images_root = images_root.resolve()
        self.augmentations = augmentations
        self.summary = json.loads(
            self.index.connection.execute(
                "SELECT value FROM metadata WHERE key='summary'"
            ).fetchone()[0]
        )
        self.split = str(self.summary["name"])
        self.source = str(self.summary["source"])
        if self.source not in ("synthetic", "real"):
            self.close()
            raise ValueError("Unsupported dataset source")

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        index_dir: Path,
        split: str,
        augmentations: Augmentations | None = None,
    ) -> "FilamentDataset":
        if split not in config["datasets"]:
            raise ValueError(f"Unknown dataset split: {split}")
        dataset = cls(
            index_dir / f"{split}.sqlite", Path(config["datasets"][split]["images"]), augmentations
        )
        if dataset.split != split or dataset.source != config["datasets"][split]["source"]:
            dataset.close()
            raise ValueError("Index source/split does not match the selected dataset config")
        return dataset

    def __enter__(self) -> "FilamentDataset":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.index.close()

    def __len__(self) -> int:
        return int(
            self.index.connection.execute(
                "SELECT COUNT(*) FROM physical_frames WHERE valid=1 AND conflicted=0 "
                "AND sha256 IS NOT NULL"
            ).fetchone()[0]
        )

    def __iter__(self) -> Iterator[FrameSample]:
        for (identifier,) in self.index.connection.execute(
            "SELECT p.canonical_image_id FROM physical_frames p JOIN images i "
            "ON i.id=p.canonical_image_id WHERE p.valid=1 AND p.conflicted=0 "
            "AND p.sha256 IS NOT NULL ORDER BY i.sequence_id,i.frame_index,p.file_name"
        ):
            yield self.get_frame(identifier)

    def get_frame(self, image_id: str | int) -> FrameSample:
        identifier = exact_id(image_id)
        row = self.index.connection.execute(
            "SELECT p.canonical_image_id,p.file_name,p.valid,p.conflicted,p.sha256 "
            "FROM images i JOIN physical_frames p ON p.file_name=i.file_name WHERE i.id=?",
            (identifier,),
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        canonical, filename, valid, conflicted, digest = row
        if not valid or conflicted or digest is None:
            raise ValueError(
                f"Frame {identifier} is invalid, conflicted or unreadable in audit cache"
            )
        raw = self.index.get_frame(canonical)
        info = raw["image"]
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Image path escapes the configured image root")
        path = self.images_root / relative
        if len(relative.parts) > 1 or path.is_symlink():
            path = path.resolve()
            if not path.is_relative_to(self.images_root):
                raise ValueError("Image path escapes the configured image root")
        png_bytes = path.read_bytes()
        if hashlib.sha256(png_bytes).hexdigest() != digest:
            raise ValueError("PNG contents changed since audit; create a new audited data version")
        with Image.open(BytesIO(png_bytes)) as image:
            if image.mode not in ("L", "RGB", "RGBA", "P", "1"):
                raise ValueError("Dataset API requires 8-bit images")
            if image.size != (info["width"], info["height"]):
                raise ValueError("PNG dimensions changed since audit")
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
        source_ids = tuple(
            row[0]
            for row in self.index.connection.execute(
                "SELECT id FROM images WHERE file_name=? ORDER BY rowid", (filename,)
            )
        )
        sequence_id = exact_id(info["sequence_id"]) if self.source == "synthetic" else None
        frame_index = int(Path(filename).stem.rsplit("_", 1)[-1])
        geometry = GeometryTrace((info["height"], info["width"]))
        instances = []
        for annotation in raw["annotations"]:
            bbox = tuple(float(v) for v in annotation["bbox"])
            if len(bbox) != 4 or not np.isfinite(bbox).all() or min(bbox[2:]) < 0:
                raise ValueError(f"Invalid source bbox for instance {annotation['id']}")
            box = (bbox[0], bbox[1], bbox[2], bbox[3])
            mask = LazyMask(annotation["segmentation"], geometry)
            instances.append(
                InstanceSample(
                    exact_id(annotation["id"]),
                    exact_id(annotation["category_id"]),
                    exact_id(annotation["track_id"])
                    if annotation.get("track_id") is not None
                    else None,
                    box,
                    box,
                    mask,
                    bool(annotation.get("iscrowd", 0)),
                )
            )
        metadata = FrameMetadata(
            self.split,
            self.source,
            exact_id(canonical),
            filename,
            source_ids,
            sequence_id,
            frame_index,
            self.source == "synthetic",
        )
        result = FrameSample(
            grayscale, np.ones(grayscale.shape, dtype=bool), tuple(instances), metadata, geometry
        )
        return self.augmentations.apply(result) if self.augmentations else result

    def sequence_ids(self) -> Iterator[str]:
        if self.source != "synthetic":
            raise ValueError("Real window IDs are not confirmed independent temporal sequences")
        for (sequence,) in self.index.connection.execute(
            "SELECT DISTINCT sequence_id FROM images WHERE sequence_id IS NOT NULL "
            "ORDER BY sequence_id"
        ):
            yield str(sequence)

    def get_sequence(self, sequence_id: str | int) -> Iterator[FrameSample]:
        if self.source != "synthetic":
            raise ValueError(
                "Real overlapping windows are not videos; use get_window for inspection"
            )
        found = False
        size = None
        for (identifier,) in self.index.connection.execute(
            "SELECT id FROM images WHERE sequence_id=? ORDER BY frame_index,id",
            (exact_id(sequence_id),),
        ):
            frame = self.get_frame(identifier)
            if size is not None and frame.geometry.original_size != size:
                raise ValueError("Temporal frames must have identical original dimensions")
            size, found = frame.geometry.original_size, True
            yield frame
        if not found:
            raise KeyError(sequence_id)

    def get_window(self, window_id: str | int) -> Iterator[FrameSample]:
        if self.source != "real":
            raise ValueError("Window inspection is only available for real data")
        found = False
        window = exact_id(window_id)
        for (identifier,) in self.index.connection.execute(
            "SELECT id FROM images WHERE sequence_id=? ORDER BY frame_index,id", (window,)
        ):
            found = True
            frame = self.get_frame(identifier)
            yield replace(frame, metadata=replace(frame.metadata, window_id=window))
        if not found:
            raise KeyError(window)
