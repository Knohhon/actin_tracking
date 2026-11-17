"""Geometry, temporal semantics, lazy decode and preview integration on tiny fixtures."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from actin_tracking.cli import main, resolve_config
from actin_tracking.data.audit import audit_dataset
from actin_tracking.data.augmentations import Augmentations
from actin_tracking.data.dataset import FilamentDataset
from actin_tracking.data.geometry import GeometryOp, GeometryTrace
from actin_tracking.data.rle import decode_rle, encode_rle
from actin_tracking.data.samples import collate_frames, transform_frame
from actin_tracking.visualization.inspector import Trajectories, identity_color, render_frame

BIG = 30909304266160180
SEQ = str(BIG + 10)


def fixture(
    root: Path, *, real: bool = False, indices: tuple[int, ...] = (0, 1, 2), mismatch: bool = False
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    pngs = root / "pngs"
    pngs.mkdir()
    split = "real_eval" if real else "synthetic_val"
    images, annotations = [], []
    actual_mask = np.zeros((7, 9), dtype=np.uint8)
    actual_mask[2:5, 3:7] = 1
    for position, number in enumerate((100, 101, 102, 101, 102, 103) if real else indices):
        filename = f"{number}.png" if real else f"{SEQ}_{number:03d}.png"
        pixels = np.full(actual_mask.shape, 20, dtype=np.uint8)
        pixels[actual_mask > 0] = 240
        Image.fromarray(pixels).save(pngs / filename)
        image_id = BIG + 100 + position
        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": 9,
                "height": 7,
                "sequence_id": (10 if position < 3 else 11) if real else int(SEQ),
            }
        )
        annotations.append(
            {
                "id": BIG + 1000 + position,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [3, 2, 3, 2] if mismatch else [3, 2, 4, 3],
                "area": 12,
                "iscrowd": 0,
                "segmentation": encode_rle(actual_mask),
                **({} if real else {"track_id": BIG + 20}),
            }
        )
    source = root / "source.json"
    source.write_text(
        json.dumps(
            {
                "images": list(reversed(images)),
                "annotations": annotations,
                "categories": [{"id": 1, "name": "filament"}],
            }
        )
    )
    output = root / "index"
    output.mkdir()
    audit_dataset(
        split,
        {
            "source": "real" if real else "synthetic",
            "images": str(pngs),
            "annotations": str(source),
        },
        output,
        mask_sample=0,
    )
    return output / f"{split}.sqlite", pngs


def test_dataset_is_lazy_and_preserves_exact_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, pngs = fixture(tmp_path)
    with FilamentDataset(index, pngs) as dataset:

        def unexpected_decode(*args: Any, **kwargs: Any) -> None:
            pytest.fail("Loading frame metadata must not decode masks")

        with monkeypatch.context() as patcher:
            patcher.setattr("actin_tracking.data.samples.decode_rle", unexpected_decode)
            frame = dataset.get_frame(BIG + 100)
            assert len(dataset) == 3
            assert frame.image.shape == (7, 9)
            assert frame.image.dtype == np.uint8
            assert frame.metadata.sequence_id == SEQ
            assert frame.metadata.image_id == str(BIG + 100)
            assert frame.instances[0].track_id == str(BIG + 20)
            assert frame.metadata.fps is None and frame.metadata.pixel_size_um is None
        assert frame.instances[0].mask.decode().sum() == 12
        assert str(BIG + 20) in json.dumps(frame.instances[0].track_id)


def test_numeric_order_independent_of_json_order(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path, indices=(10, 0, 2))
    with FilamentDataset(index, pngs) as dataset:
        assert [f.metadata.frame_index for f in dataset.get_sequence(SEQ)] == [0, 2, 10]
        assert [f.metadata.frame_index for f in dataset] == [0, 2, 10]
        assert list(dataset.sequence_ids()) == [SEQ]
        with pytest.raises(KeyError):
            list(dataset.get_sequence("unknown"))


def test_real_iteration_deduplicates_windows_and_rejects_temporal_use(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path, real=True)
    with FilamentDataset(index, pngs) as dataset:
        assert len(dataset) == 4
        assert [f.metadata.file_name for f in dataset] == [
            "100.png",
            "101.png",
            "102.png",
            "103.png",
        ]
        window = list(dataset.get_window(10))
        assert len(window) == 3
        assert all(not f.metadata.temporal_continuity_confirmed for f in window)
        assert all(f.metadata.sequence_id is None and f.metadata.window_id == "10" for f in window)
        frame = dataset.get_frame(BIG + 103)
        assert frame.metadata.file_name == "101.png"
        assert len(frame.metadata.source_image_ids) == 2
        with pytest.raises(ValueError, match="not videos"):
            list(dataset.get_sequence(10))
        with pytest.raises(ValueError):
            list(dataset.sequence_ids())
        with pytest.raises(ValueError, match="Duplicate physical"):
            collate_frames([frame, dataset.get_frame(BIG + 101)])
        assert collate_frames([frame, frame], allow_duplicates=True).images.shape[0] == 2


def test_rle_compressed_uncompressed_and_native_roundtrip() -> None:
    expected = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    assert np.array_equal(decode_rle(encode_rle(expected)), expected)
    native = coco_mask.encode(np.asfortranarray(expected))
    assert np.array_equal(decode_rle(native), expected)
    assert np.array_equal(decode_rle({"size": [2, 3], "counts": [1, 2, 1, 1, 1]}), expected)
    with pytest.raises(ValueError):
        decode_rle({"size": [2, 3], "counts": [5]})
    with pytest.raises(ValueError):
        encode_rle(np.full((2, 2), 2, dtype=np.uint8))


def test_resize_pad_crop_flip_and_inverse_coordinates(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path)
    ops = (
        GeometryOp("resize", (14, 18)),
        GeometryOp("pad", (3, 2, 4, 5)),
        GeometryOp("crop", (2, 1, 21, 18)),
        GeometryOp("hflip"),
        GeometryOp("vflip"),
    )
    with FilamentDataset(index, pngs) as dataset:
        original = dataset.get_frame(BIG + 100)
        frame = transform_frame(original, ops)
        assert frame.image.shape == (18, 21)
        assert frame.instances[0].bbox_xywh == (6, 7, 8, 6)
        expected_mask = np.zeros((18, 21), dtype=np.uint8)
        expected_mask[7:13, 6:14] = 1
        assert np.array_equal(frame.instances[0].mask.decode(), expected_mask)
        assert np.array_equal(decode_rle(frame.instances[0].mask.to_rle()), expected_mask)
        points = np.array([[0.5, 0.5], [3.5, 2.5], [8.5, 6.5]])
        assert np.allclose(
            frame.geometry.points(frame.geometry.points(points), inverse=True), points
        )
        boxes = np.array([[3, 2, 4, 3]], dtype=float)
        assert np.allclose(frame.geometry.boxes(frame.geometry.boxes(boxes), inverse=True), boxes)
        restored, visible = frame.geometry.restore_mask(expected_mask)
        assert visible.all()
        assert np.array_equal(restored, original.instances[0].mask.decode())
        assert original.image.shape == (7, 9) and not original.geometry.steps


def test_crop_reports_lost_information_and_removed_instances(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path)
    with FilamentDataset(index, pngs) as dataset:
        original = dataset.get_frame(BIG + 100)
        frame = transform_frame(original, (GeometryOp("crop", (4, 3, 4, 3)),))
        assert frame.instances[0].bbox_xywh == (0, 0, 3, 2)
        assert frame.instances[0].original_bbox_xywh == (3, 2, 4, 3)
        restored, visible = frame.geometry.restore_mask(frame.instances[0].mask.decode())
        roi = np.zeros((7, 9), dtype=bool)
        roi[3:6, 4:8] = True
        assert np.array_equal(visible, roi)
        assert np.array_equal(restored, original.instances[0].mask.decode() * roi)
        hidden = transform_frame(original, (GeometryOp("crop", (0, 0, 2, 2)),))
        assert hidden.instances == ()
        assert hidden.dropped_instance_ids == (original.instances[0].instance_id,)


def test_sequence_augmentations_are_deterministic_and_share_parameters(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path)
    augment = Augmentations(
        seed=73,
        resize=(14, 18),
        crop_size=(12, 15),
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.5,
        brightness=(0.9, 1.1),
        noise_std=2,
    )
    with FilamentDataset(index, pngs, augment) as dataset:
        frames = list(dataset.get_sequence(SEQ))
        assert all(f.geometry.as_dict() == frames[0].geometry.as_dict() for f in frames)
        assert all(f.photometric_history == frames[0].photometric_history for f in frames)
        same = dataset.get_frame(BIG + 100)
        assert np.array_equal(same.image, frames[0].image)
        assert np.array_equal(same.instances[0].mask.decode(), frames[0].instances[0].mask.decode())
        assert np.array_equal(same.image, dataset.get_frame(BIG + 100).image)


def test_batch_padding_tracks_coordinates_validity_and_targets(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path)
    with FilamentDataset(index, pngs) as dataset:
        first = dataset.get_frame(BIG + 100)
        second = transform_frame(dataset.get_frame(BIG + 101), (GeometryOp("pad", (2, 1, 0, 0)),))
        batch = collate_frames([first, second])
        assert batch.images.shape == (2, 8, 11)
        assert batch.sizes_before_batch == ((7, 9), (8, 11))
        assert batch.valid_pixels.sum(axis=(1, 2)).tolist() == [63, 63]
        assert not batch.valid_pixels[0, 7].any() and not batch.valid_pixels[1, :, :2].any()
        assert batch.frames[1].instances[0].bbox_xywh == (5, 3, 4, 3)
        restored, visible = batch.frames[1].geometry.restore_mask(
            batch.frames[1].instances[0].mask.decode()
        )
        assert visible.all() and restored.sum() == 12
        with pytest.raises(ValueError):
            collate_frames([])


def test_source_box_discrepancy_is_preserved_and_mask_view_is_aligned(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path, mismatch=True)
    with FilamentDataset(index, pngs) as dataset:
        frame = dataset.get_frame(BIG + 100)
        assert frame.instances[0].bbox_xywh == (3, 2, 3, 2)
        assert frame.instances[0].mask.bbox_xywh() == (3, 2, 4, 3)
        rendered, record = render_frame(frame, ["masks"])
        rgb = np.asarray(rendered)
        foreground = frame.instances[0].mask.decode().astype(bool)
        assert np.array_equal(
            rgb[~foreground], np.repeat(frame.image[:, :, None], 3, 2)[~foreground]
        )
        assert not np.array_equal(
            rgb[foreground], np.repeat(frame.image[:, :, None], 3, 2)[foreground]
        )
        assert record["instances"][0]["mask_bbox_xywh"] == [3, 2, 4, 3]


def test_source_image_change_after_audit_is_detected(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path)
    Image.new("L", (9, 7), 99).save(pngs / f"{SEQ}_000.png")
    with FilamentDataset(index, pngs) as dataset:
        with pytest.raises(ValueError, match="changed since audit"):
            dataset.get_frame(BIG + 100)


def test_trajectory_gaps_and_unknown_real_identity_are_not_interpolated(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path, indices=(0, 2, 3))
    with FilamentDataset(index, pngs) as dataset:
        state = Trajectories()
        frames = list(dataset.get_sequence(SEQ))
        render_frame(frames[0], ["trajectories"], state)
        _, record = render_frame(frames[1], ["trajectories"], state)
        points = next(iter(state.points.values()))
        assert [p[0] for p in points] == [0, 2]
        from unittest.mock import Mock

        drawing = Mock()
        state.draw(drawing)
        drawing.line.assert_not_called()
        render_frame(frames[2], ["trajectories"], state)
        state.draw(drawing)
        drawing.line.assert_called_once()
        shifted = transform_frame(frames[2], (GeometryOp("pad", (1, 0, 0, 0)),))
        shifted = replace(shifted, metadata=replace(shifted.metadata, frame_index=4))
        render_frame(shifted, ["trajectories"], state)
        assert all(len(points) == 1 for points in state.points.values())
        # Isolated centroid has no segment joining the skipped frame.
        assert len(record["instances"]) == 1
        assert identity_color("x") == identity_color("x")
        real = replace(
            frames[0],
            metadata=replace(
                frames[0].metadata,
                source="real",
                sequence_id=None,
                temporal_continuity_confirmed=False,
            ),
        )
        with pytest.raises(ValueError, match="confirmed"):
            render_frame(real, ["trajectories"], state)


def test_invalid_transform_arguments_are_rejected() -> None:
    with pytest.raises(ValueError):
        GeometryTrace((7, 9)).append((GeometryOp("crop", (8, 0, 4, 2)),))
    with pytest.raises(ValueError):
        GeometryTrace((7, 9)).append((GeometryOp("resize", (0, 9)),))
    with pytest.raises(ValueError):
        Augmentations(horizontal_flip_probability=2)


def write_config(root: Path, pngs: Path) -> Path:
    path = root / "datasets.toml"
    path.write_text(
        'schema_version = "1.0"\nroot = "."\n'
        + "\n".join(
            f'[datasets.{name}]\nsource = "{source}"\nimages = "{pngs.name}"\n'
            'annotations = "source.json"\n'
            for name, source in (
                ("synthetic_train", "synthetic"),
                ("synthetic_val", "synthetic"),
                ("real_eval", "real"),
            )
        )
    )
    return path


def test_visualize_cli_dry_run_export_and_exact_ids(tmp_path: Path) -> None:
    index, pngs = fixture(tmp_path)
    config = write_config(tmp_path, pngs)
    output = tmp_path / "preview"
    args = [
        "visualize",
        "--config",
        str(config),
        "--index-dir",
        str(index.parent),
        "--dataset",
        "synthetic_val",
        "--sequence-id",
        SEQ,
        "--output",
        str(output),
        "--limit",
        "2",
        "--layers",
        "masks",
        "boxes",
        "skeleton",
        "tracks",
        "trajectories",
        "--resize",
        "14",
        "18",
        "--hflip",
    ]
    assert main([*args, "--dry-run"]) == 0
    assert not output.exists()
    assert main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["frames"]) == 2
    assert (
        manifest["config_sha256"]
        == hashlib.sha256((output / "config.resolved.json").read_bytes()).hexdigest()
    )
    prefix = manifest["frames"][0]["prefix"]
    record = json.loads((output / f"{prefix}.json").read_text())
    assert record["metadata"]["image_id"] == str(BIG + 100)
    assert record["instances"][0]["track_id"] == str(BIG + 20)
    assert Image.open(output / f"{prefix}_comparison.png").size == (36, 42)
    assert len(record["geometry"]["steps"]) == 2
    with pytest.raises(SystemExit) as failure:
        main(args)
    assert failure.value.code == 2
    with FilamentDataset.from_config(
        resolve_config(config, None), index.parent, "synthetic_val"
    ) as data:
        assert len(data) == 3
    with pytest.raises(ValueError):
        FilamentDataset.from_config(resolve_config(config, None), index.parent, "unknown")
    import shutil

    shutil.copyfile(index, index.parent / "synthetic_train.sqlite")
    with pytest.raises(ValueError, match="source/split"):
        FilamentDataset.from_config(resolve_config(config, None), index.parent, "synthetic_train")


def test_skeleton_keeps_small_foreground_and_original_position() -> None:
    from actin_tracking.visualization.inspector import skeleton_mask

    mask = np.zeros((11, 13), dtype=np.uint8)
    mask[4:6, 7:9] = 1
    skeleton = skeleton_mask(mask)
    assert skeleton.any()
    assert np.all(~skeleton | mask.astype(bool))
    assert not skeleton_mask(np.zeros_like(mask)).any()


def test_temporal_collate_pads_time_and_forbids_real_windows_and_missing_frames(
    tmp_path: Path,
) -> None:
    from actin_tracking.data.samples import collate_sequences

    index, pngs = fixture(tmp_path)
    with FilamentDataset(index, pngs) as dataset:
        sequence = list(dataset.get_sequence(SEQ))
        batch = collate_sequences([sequence[:2], sequence], allow_duplicates=True)
        assert batch.images.shape == (2, 3, 7, 9)
        assert batch.valid_frames.tolist() == [[True, True, False], [True, True, True]]
        assert not batch.valid_pixels[0, 2].any()
        assert batch.sequences[0][0].instances[0].track_id == str(BIG + 20)
        with pytest.raises(ValueError, match="Duplicate"):
            collate_sequences([sequence, sequence])
        with pytest.raises(ValueError, match="consecutive"):
            collate_sequences([[sequence[0], sequence[2]]])
        real = replace(
            sequence[0],
            metadata=replace(
                sequence[0].metadata,
                sequence_id=None,
                temporal_continuity_confirmed=False,
                source="real",
            ),
        )
        with pytest.raises(ValueError, match="confirmed"):
            collate_sequences([[real]])
        different_geometry = transform_frame(sequence[1], (GeometryOp("hflip"),))
        with pytest.raises(ValueError, match="transformations"):
            collate_sequences([[sequence[0], different_geometry]])


def test_iteration_opens_only_requested_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    index, pngs = fixture(tmp_path)
    with FilamentDataset(index, pngs) as dataset:
        original = dataset.get_frame
        calls = Mock(wraps=original)
        monkeypatch.setattr(dataset, "get_frame", calls)
        iterator = iter(dataset)
        calls.assert_not_called()
        next(iterator)
        assert calls.call_count == 1
        temporal = dataset.get_sequence(SEQ)
        assert calls.call_count == 1
        next(temporal)
        assert calls.call_count == 2
