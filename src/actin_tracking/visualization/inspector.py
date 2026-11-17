"""Render source GT and transformed geometry; never fabricate real trajectories."""

import hashlib
import importlib.metadata
import json
import platform
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

from actin_tracking.data.dataset import FilamentDataset
from actin_tracking.data.geometry import GeometryTrace
from actin_tracking.data.samples import FrameSample

LAYERS = frozenset({"masks", "boxes", "skeleton", "tracks", "trajectories"})


def identity_color(identity: str) -> tuple[int, int, int]:
    raw = hashlib.sha256(identity.encode()).digest()
    return (64 + raw[0] % 192, 64 + raw[1] % 192, 64 + raw[2] % 192)


def skeleton_mask(mask: NDArray[Any]) -> NDArray[np.bool_]:
    """Skeletonize only the foreground ROI; retain original image coordinates."""
    binary = np.asarray(mask, dtype=bool)
    output = np.zeros_like(binary)
    yy, xx = np.nonzero(binary)
    if not len(xx):
        return output
    top, bottom, left, right = int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1
    roi = np.pad(binary[top:bottom, left:right], 1)
    thinned = skeletonize(roi, method="zhang")  # type: ignore[no-untyped-call]
    skeleton = np.asarray(thinned, dtype=np.bool_)[1:-1, 1:-1]
    output[top:bottom, left:right] = skeleton
    return output


@dataclass
class Trajectories:
    history: int = 30
    sequence: str | None = None
    previous_index: int | None = None
    geometry: GeometryTrace | None = None
    points: dict[str, deque[tuple[int, tuple[float, float]]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.history) is not int or self.history <= 0:
            raise ValueError("Trajectory history must be a positive integer")

    def update(self, frame: FrameSample, centers: dict[str, tuple[float, float]]) -> None:
        meta = frame.metadata
        if not meta.temporal_continuity_confirmed or meta.sequence_id is None:
            self.points.clear()
            self.sequence = None
            return
        if (
            self.sequence != meta.sequence_id
            or self.geometry != frame.geometry
            or (self.previous_index is not None and meta.frame_index <= self.previous_index)
        ):
            self.points.clear()
        self.sequence, self.previous_index = meta.sequence_id, meta.frame_index
        self.geometry = frame.geometry
        for identity, center in centers.items():
            self.points.setdefault(identity, deque(maxlen=self.history)).append(
                (meta.frame_index, center)
            )
        for identity in list(self.points):
            while (
                self.points[identity]
                and self.points[identity][0][0] < meta.frame_index - self.history
            ):
                self.points[identity].popleft()
            if not self.points[identity]:
                del self.points[identity]

    def draw(self, drawing: ImageDraw.ImageDraw) -> None:
        for identity, points in self.points.items():
            for (previous, p), (current, q) in zip(points, list(points)[1:], strict=False):
                if current == previous + 1:
                    drawing.line([p, q], fill=identity_color(identity), width=2)


def render_frame(
    frame: FrameSample,
    layers: Iterable[str] = ("masks", "boxes", "tracks"),
    trajectories: Trajectories | None = None,
    max_labels: int = 20,
) -> tuple[Image.Image, dict[str, Any]]:
    selected = set(layers)
    if not selected <= LAYERS or max_labels < 0:
        raise ValueError("Unknown inspector layer or invalid max_labels")
    if "trajectories" in selected and not frame.metadata.temporal_continuity_confirmed:
        raise ValueError("GT trajectories require confirmed temporal sequences")
    rgb = np.repeat(frame.image[:, :, None], 3, axis=2)
    details = []
    track_centers: dict[str, tuple[float, float]] = {}
    ambiguous: set[str] = set()
    labels: list[tuple[str, tuple[int, int, int], tuple[float, float, float, float]]] = []
    namespace = frame.metadata.sequence_id or frame.metadata.file_name
    for instance in frame.instances:
        confirmed_track = (
            instance.track_id if frame.metadata.temporal_continuity_confirmed else None
        )
        identity = f"{namespace}:{confirmed_track or instance.instance_id}"
        color = identity_color(identity)
        center = None
        mask_box = None
        if selected & {"masks", "skeleton", "trajectories"}:
            mask = instance.mask.decode().astype(bool)
            yy, xx = np.nonzero(mask)
            if len(xx):
                center = (float(xx.mean() + 0.5), float(yy.mean() + 0.5))
                mask_box = [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max() - xx.min() + 1),
                    int(yy.max() - yy.min() + 1),
                ]
                if "masks" in selected:
                    rgb[mask] = np.rint(rgb[mask] * 0.55 + np.asarray(color) * 0.45).astype(
                        np.uint8
                    )
                if "skeleton" in selected:
                    rgb[skeleton_mask(mask)] = color
                if instance.track_id is not None:
                    if identity in track_centers:
                        ambiguous.add(identity)
                    track_centers[identity] = center
        label = (
            f"t:{instance.track_id}"
            if instance.track_id is not None
            else f"i:{instance.instance_id}"
        )
        labels.append((label, color, instance.bbox_xywh))
        details.append(
            {
                "instance_id": instance.instance_id,
                "track_id": instance.track_id,
                "category_id": instance.category_id,
                "bbox_xywh": instance.bbox_xywh,
                "original_bbox_xywh": instance.original_bbox_xywh,
                "mask_bbox_xywh": mask_box,
                "visible_mask_center_xy": center,
            }
        )
    result = Image.fromarray(rgb)
    drawing = ImageDraw.Draw(result)
    if "trajectories" in selected and trajectories is not None:
        trajectories.update(frame, {k: v for k, v in track_centers.items() if k not in ambiguous})
        trajectories.draw(drawing)
    for i, (label, color, box) in enumerate(labels):
        x, y, w, h = box
        if "boxes" in selected and w > 0 and h > 0:
            drawing.rectangle([x, y, max(x, x + w - 1), max(y, y + h - 1)], outline=color, width=1)
        if "tracks" in selected and i < max_labels:
            # Keep long exact IDs within the canvas; full IDs also appear in JSON.
            bounds = drawing.textbbox((0, 0), label)
            label_x = max(0.0, min(x, frame.image.shape[1] - (bounds[2] - bounds[0])))
            label_y = max(0.0, y - 11)
            drawing.text(
                (label_x, label_y), label, fill=color, stroke_width=1, stroke_fill=(0, 0, 0)
            )
    record = {
        "metadata": asdict(frame.metadata),
        "geometry": frame.geometry.as_dict(),
        "photometric_history": frame.photometric_history,
        "dropped_instance_ids": frame.dropped_instance_ids,
        "instances": details,
        "layers": sorted(selected),
        "max_labels": max_labels,
        "ambiguous_track_ids": sorted(ambiguous),
        "skeleton_method": "skimage skeletonize, zhang, padded foreground ROI",
        "centroid_definition": "Visible foreground pixel centers; gaps are not interpolated",
    }
    return result, record


def choose_frames(
    dataset: FilamentDataset,
    *,
    image_id: str | None = None,
    sequence_id: str | None = None,
    window_id: str | None = None,
    seed: int = 20261117,
    limit: int = 3,
) -> Iterator[FrameSample]:
    if limit <= 0:
        raise ValueError("Preview limit must be positive")
    if sum(value is not None for value in (image_id, sequence_id, window_id)) > 1:
        raise ValueError("Select only one image, sequence, or window")
    if image_id is not None:
        yield dataset.get_frame(image_id)
    elif sequence_id is not None:
        yield from islice(dataset.get_sequence(sequence_id), limit)
    elif window_id is not None:
        yield from islice(dataset.get_window(window_id), limit)
    else:
        dataset.index.connection.create_function(
            "preview_key", 1, lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        )
        for (identifier,) in dataset.index.connection.execute(
            "SELECT canonical_image_id FROM physical_frames WHERE valid=1 AND conflicted=0 "
            "AND sha256 IS NOT NULL ORDER BY preview_key(file_name),file_name LIMIT ?",
            (limit,),
        ):
            yield dataset.get_frame(identifier)


def visualize(
    dataset: FilamentDataset,
    output: Path,
    *,
    layers: Iterable[str],
    selection: dict[str, Any],
    config: dict[str, Any],
    max_labels: int = 20,
) -> dict[str, Any]:
    selected = set(layers)
    if not selected <= LAYERS or max_labels < 0:
        raise ValueError("Unknown inspector layer or invalid max_labels")
    if "trajectories" in selected and dataset.source != "synthetic":
        raise ValueError("Real GT trajectories are unavailable; windows are inspection only")
    if "trajectories" in selected and selection.get("sequence_id") is None:
        raise ValueError("Select --sequence-id for ordered GT trajectory inspection")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "config": config,
        "selection": selection,
        "layers": sorted(selected),
        "max_labels": max_labels,
    }
    (output / "config.resolved.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "running",
        "split": dataset.split,
        "annotation_sha256": dataset.summary["annotation_sha256"],
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pillow", "pycocotools", "scikit-image")
        },
        "code_sha256": {
            str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(__file__).parents[1].rglob("*.py"))
        },
        "frames": [],
        "config_sha256": hashlib.sha256((output / "config.resolved.json").read_bytes()).hexdigest(),
        "bbox_policy": "Source boxes preserved and transformed; mask-derived boxes are separate",
        "real_policy": "Unique physical frames; windows are not independent temporal sequences",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    trajectories = Trajectories()
    try:
        for number, frame in enumerate(choose_frames(dataset, **selection)):
            image, record = render_frame(frame, selected, trajectories, max_labels)
            key = hashlib.sha256(frame.metadata.image_id.encode()).hexdigest()[:12]
            prefix = f"{number:04d}_{key}"
            Image.fromarray(frame.image).save(output / f"{prefix}_raw.png")
            image.save(output / f"{prefix}_overlay.png")
            height, width = frame.geometry.size
            comparison = Image.new("RGB", (2 * width, height + 28), (20, 20, 20))
            comparison.paste(Image.fromarray(frame.image).convert("RGB"), (0, 28))
            comparison.paste(image, (width, 28))
            draw = ImageDraw.Draw(comparison)
            draw.text((4, 5), f"Raw | {frame.metadata.file_name}", fill="white")
            draw.text(
                (width + 4, 5),
                f"GT: {len(frame.instances)} instances; labels <= {max_labels}",
                fill="white",
            )
            comparison.save(output / f"{prefix}_comparison.png")
            (output / f"{prefix}.json").write_text(json.dumps(record, indent=2) + "\n")
            manifest["frames"].append(
                {
                    "image_id": frame.metadata.image_id,
                    "file_name": frame.metadata.file_name,
                    "prefix": prefix,
                }
            )
        if not manifest["frames"]:
            raise ValueError("No eligible frames available for inspection")
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
