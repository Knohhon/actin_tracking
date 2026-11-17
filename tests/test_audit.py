"""Integration fixtures exercise the source format, not the production dataset."""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as mask_utils

from actin_tracking.data.audit import audit_dataset

LARGE_ID = 30909304266160180


def encoded_mask() -> dict[str, Any]:
    mask = np.zeros((8, 8), dtype=np.uint8, order="F")
    mask[2:4, 3:5] = 1
    rle = mask_utils.encode(mask)
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def annotation(identifier: int, image_id: int) -> dict[str, Any]:
    return {
        "id": identifier,
        "image_id": image_id,
        "category_id": 1,
        "track_id": LARGE_ID,
        "iscrowd": 0,
        "segmentation": encoded_mask(),
        "bbox": [3, 2, 2, 2],
        "area": 4,
    }


def write_dataset(
    root: Path, images: list[dict[str, Any]], annotations: list[dict[str, Any]], source: str
) -> dict[str, Any]:
    pngs = root / "pngs"
    pngs.mkdir()
    for image in images:
        Image.new("L", (8, 8), color=20).save(pngs / image["file_name"])
    payload = {
        "info": {"version": "fixture"},
        "licenses": [{"id": 1, "name": "fixture only"}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "filament"}],
    }
    path = root / "source.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"source": source, "images": str(pngs), "annotations": str(path)}


def synthetic_fixture(tmp_path: Path) -> dict[str, Any]:
    # Intentionally reverse the COCO order, including the 2/10 lexical trap.
    images = [
        {
            "id": LARGE_ID + index,
            "file_name": f"123_{index}.png",
            "sequence_id": 123,
            "height": 8,
            "width": 8,
        }
        for index in reversed(range(30))
    ]
    annotations = [
        annotation(LARGE_ID + 100 + index, image["id"]) for index, image in enumerate(images)
    ]
    return write_dataset(tmp_path, images, annotations, "synthetic")


def run_audit(tmp_path: Path, dataset: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    output = tmp_path / "audit"
    output.mkdir()
    return audit_dataset("fixture", dataset, output, **kwargs)


def test_streaming_import_preserves_source_and_exact_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = synthetic_fixture(tmp_path)
    source = Path(dataset["annotations"])
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    def no_json_load(*args: Any, **kwargs: Any) -> None:
        pytest.fail("The source must be streamed, not loaded through json.load")

    monkeypatch.setattr(json, "load", no_json_load)
    summary = run_audit(tmp_path, dataset, full_masks=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert summary["annotation_sha256"] == original_hash
    assert (tmp_path / "audit/fixture.sqlite").is_file()
    # Also serialize the report: Decimal or raw bytes must not leak into artifacts.
    json.dumps(summary)


def test_corrupt_json_does_not_look_like_a_successful_audit(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text('{"images": [', encoding="utf-8")
    dataset = {"source": "synthetic", "images": str(tmp_path), "annotations": str(source)}
    with pytest.raises((ValueError, RuntimeError)):
        run_audit(tmp_path, dataset)


def real_fixture(tmp_path: Path, conflict: bool = False) -> dict[str, Any]:
    images = [
        {"id": 1, "file_name": "100.png", "height": 8, "width": 8, "sequence_id": 10},
        {"id": 2, "file_name": "101.png", "height": 8, "width": 8, "sequence_id": 10},
        {"id": 3, "file_name": "101.png", "height": 8, "width": 8, "sequence_id": 11},
        {"id": 4, "file_name": "102.png", "height": 8, "width": 8, "sequence_id": 11},
    ]
    annotations = [annotation(100 + image["id"], image["id"]) for image in images]
    for item in annotations:
        item.pop("track_id")
    if conflict:
        mask = np.zeros((8, 8), dtype=np.uint8, order="F")
        mask[4:6, 3:5] = 1
        rle = mask_utils.encode(mask)
        annotations[2]["segmentation"] = {
            "size": list(rle["size"]),
            "counts": rle["counts"].decode("ascii"),
        }
        annotations[2]["bbox"] = [3, 4, 2, 2]
    return write_dataset(tmp_path, images, annotations, "real")


def invalid_fixture(tmp_path: Path) -> dict[str, Any]:
    images = [{"id": 1, "file_name": "123_000.png", "sequence_id": 123, "height": 8, "width": 8}]
    first = annotation(LARGE_ID, 1)
    second = annotation(LARGE_ID + 1, 999)
    third = annotation(LARGE_ID + 2, 1)
    third["category_id"] = 999
    third["segmentation"] = {"size": [8, 8], "counts": "invalid!"}
    third["bbox"] = [-1, 2, 200, 2]
    return write_dataset(
        tmp_path, images, [first, second, third, copy.deepcopy(first)], "synthetic"
    )


def test_synthetic_index_orders_frames_and_preserves_ids(tmp_path: Path) -> None:
    dataset = synthetic_fixture(tmp_path)
    summary = run_audit(tmp_path, dataset, full_masks=True)
    assert summary["counts"]["image_records"] == 30
    assert summary["counts"]["annotations"] == 30
    assert not summary["issue_counts"]
    with sqlite3.connect(tmp_path / "audit/fixture.sqlite") as db:
        frames = db.execute(
            "SELECT id,frame_index FROM images WHERE sequence_id='123' ORDER BY frame_index"
        ).fetchall()
        assert frames == [(str(LARGE_ID + index), index) for index in range(30)]
        assert db.execute("SELECT DISTINCT track_id FROM annotations").fetchall() == [
            (str(LARGE_ID),)
        ]
    assert (
        summary["annotation_sha256"]
        == hashlib.sha256(Path(dataset["annotations"]).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("conflict", [False, True])
def test_real_window_dedup_and_conflicts(tmp_path: Path, conflict: bool) -> None:
    summary = run_audit(tmp_path, real_fixture(tmp_path, conflict=conflict), full_masks=True)
    assert summary["counts"]["image_records"] == 4
    assert summary["counts"]["unique_frames"] == 3
    assert bool(summary["issue_counts"].get("conflicting_real_annotations")) is conflict
    with sqlite3.connect(tmp_path / "audit/fixture.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM physical_frames").fetchone()[0] == 3
        assert db.execute(
            "SELECT conflicted FROM physical_frames WHERE file_name='101.png'"
        ).fetchone()[0] == int(conflict)


def test_invalid_references_and_duplicate_ids_are_reported(tmp_path: Path) -> None:
    dataset = invalid_fixture(tmp_path)
    path = Path(dataset["annotations"])
    payload = json.loads(path.read_text())
    duplicate = copy.deepcopy(payload["annotations"][0])
    duplicate["segmentation"] = {"size": [8, 8], "counts": ""}
    payload["annotations"].append(duplicate)
    path.write_text(json.dumps(payload))
    summary = run_audit(tmp_path, dataset, full_masks=True)
    issues = summary["issue_counts"]
    assert issues["invalid_image_ref"] == 1
    assert issues["invalid_category_ref"] == 1
    assert issues["duplicate_annotation_id"] >= 1
    assert issues["incomplete_sequence"] == 1
    assert issues.get("invalid_rle", 0) + issues.get("rle_decode_error", 0) >= 1


def test_sections_need_not_be_in_coco_order(tmp_path: Path) -> None:
    dataset = synthetic_fixture(tmp_path)
    path = Path(dataset["annotations"])
    payload = json.loads(path.read_text())
    path.write_text(json.dumps({key: payload[key] for key in reversed(payload)}))
    summary = run_audit(tmp_path, dataset)
    assert not summary["issue_counts"]


def test_empty_sections_and_truncated_json(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, [], [], "real")
    summary = run_audit(tmp_path, dataset)
    assert summary["issue_counts"]["empty_or_missing_section"] >= 2
    Path(dataset["annotations"]).write_text('{"images": [')
    output = tmp_path / "truncated"
    output.mkdir()
    with pytest.raises(ValueError):
        audit_dataset("broken", dataset, output)


def test_missing_png_and_declared_geometry_are_not_silently_fixed(tmp_path: Path) -> None:
    dataset = real_fixture(tmp_path)
    (Path(dataset["images"]) / "102.png").unlink()
    path = Path(dataset["annotations"])
    payload = json.loads(path.read_text())
    payload["images"][0]["width"] = 9
    payload["annotations"][0]["area"] = 7
    payload["annotations"][0]["bbox"] = [3, 2, 3, 2]
    path.write_text(json.dumps(payload))
    summary = run_audit(tmp_path, dataset, full_masks=True)
    issues = summary["issue_counts"]
    assert issues["missing_or_unreadable_png"] == 1
    assert issues["image_dimensions_mismatch"] == 1
    assert issues["mask_area_mismatch"] == 1
    assert issues["mask_bbox_mismatch"] == 1
    with sqlite3.connect(tmp_path / "audit/fixture.sqlite") as db:
        cached = json.loads(
            db.execute("SELECT payload FROM annotations WHERE id='101'").fetchone()[0]
        )
        assert cached["area"] == 7
        assert cached["bbox"] == [3, 2, 3, 2]


def test_track_identity_may_recur_but_not_twice_in_one_frame(tmp_path: Path) -> None:
    dataset = synthetic_fixture(tmp_path)
    path = Path(dataset["annotations"])
    payload = json.loads(path.read_text())
    # Remove one observation from the middle and duplicate a different observation.
    payload["annotations"] = [
        item for item in payload["annotations"] if item["image_id"] != LARGE_ID + 15
    ]
    duplicate = copy.deepcopy(payload["annotations"][0])
    duplicate["id"] = LARGE_ID + 999
    payload["annotations"].append(duplicate)
    path.write_text(json.dumps(payload))
    summary = run_audit(tmp_path, dataset)
    assert summary["issue_counts"]["duplicate_track_in_frame"] >= 1
    assert summary["issue_counts"]["track_temporal_gap"] == 1


def test_index_reader_queries_and_conflict_exclusion(tmp_path: Path) -> None:
    from actin_tracking.data.indexing import DatasetIndex

    dataset = real_fixture(tmp_path, conflict=True)
    run_audit(tmp_path, dataset)
    with DatasetIndex(tmp_path / "audit/fixture.sqlite") as index:
        frames = list(index.iter_unique_frames())
        assert len(frames) == 2
        assert {f["image"]["file_name"] for f in frames} == {"100.png", "102.png"}
        with pytest.raises(KeyError):
            index.get_frame("missing")


def test_index_reader_numeric_sequence_order(tmp_path: Path) -> None:
    from actin_tracking.data.indexing import DatasetIndex

    run_audit(tmp_path, synthetic_fixture(tmp_path))
    with DatasetIndex(tmp_path / "audit/fixture.sqlite") as index:
        frames = list(index.get_sequence(123))
        assert [f["image"]["id"] for f in frames] == [LARGE_ID + n for n in range(30)]
        assert index.get_frame(str(LARGE_ID))["annotations"][0]["track_id"] == LARGE_ID


def test_audit_cli_artifacts_and_alternative_real(tmp_path: Path) -> None:
    from actin_tracking.cli import main

    dataset = real_fixture(tmp_path)
    config = tmp_path / "datasets.toml"
    config.write_text(
        'schema_version = "1.0"\nroot = "."\n'
        + "\n".join(
            f'[datasets.{name}]\nsource = "real"\nimages = "pngs"\n'
            'annotations = "source.json"\nalternative_annotations = "source.json"\n'
            for name in ("synthetic_train", "synthetic_val", "real_eval")
        )
    )
    output = tmp_path / "cli"
    args = ["audit", "--config", str(config), "--subset", "real_eval", "--output", str(output)]
    assert main([*args, "--dry-run"]) == 0
    assert not output.exists()
    assert main(args) == 0
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert (
        manifest["datasets"][0]["annotation_sha256"]
        == hashlib.sha256(Path(dataset["annotations"]).read_bytes()).hexdigest()
    )
    assert (output / "audit_report.md").exists()
    assert (output / "real_eval_areas.svg").exists()
    assert manifest["real_annotation_comparison"]["conflicting_frames"] == []


def test_index_excludes_invalid_pngs_and_conflicting_gt(tmp_path: Path) -> None:
    from actin_tracking.data.indexing import DatasetIndex

    dataset = real_fixture(tmp_path, conflict=True)
    (Path(dataset["images"]) / "102.png").write_bytes(b"not a PNG")
    summary = run_audit(tmp_path, dataset)
    assert summary["counts"]["evaluation_frames"] == 1
    with DatasetIndex(tmp_path / "audit/fixture.sqlite") as index:
        frames = list(index.iter_unique_frames())
        assert [f["image"]["file_name"] for f in frames] == ["100.png"]
        with pytest.raises(KeyError):
            index.get_frame("missing")


def test_skeleton_diagnostic_has_explicit_pixel_definition() -> None:
    from actin_tracking.data.diagnostics import skeleton_pixels

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[4, 1:7] = 1
    assert skeleton_pixels(mask) == 6
    assert skeleton_pixels(np.zeros_like(mask)) == 0


def test_empty_rle_and_wrong_area_detected_without_decode_sample(tmp_path: Path) -> None:
    dataset = real_fixture(tmp_path)
    source = Path(dataset["annotations"])
    payload = json.loads(source.read_text())
    payload["annotations"][0]["segmentation"] = {"size": [8, 8], "counts": [64]}
    source.write_text(json.dumps(payload))
    summary = run_audit(tmp_path, dataset, mask_sample=0)
    assert summary["issue_counts"]["empty_rle_mask"] == 1
    assert summary["issue_counts"]["rle_area_mismatch"] == 1


def test_unreferenced_png_is_reported_without_becoming_evaluation_gt(tmp_path: Path) -> None:
    dataset = real_fixture(tmp_path)
    Image.new("L", (8, 8)).save(Path(dataset["images"]) / "999.png")
    summary = run_audit(tmp_path, dataset, mask_sample=0)
    assert summary["issue_counts"]["unreferenced_png"] == 1
    assert summary["counts"]["physical_frames"] == 3


def test_file_hashing_crosses_bounded_batch_and_commit_boundaries(tmp_path: Path) -> None:
    images = [{"id": n, "file_name": f"{n}.png", "width": 8, "height": 8} for n in range(257)]
    dataset = write_dataset(tmp_path, images, [], "real")
    summary = run_audit(tmp_path, dataset, mask_sample=0)
    assert summary["counts"]["evaluation_frames"] == 257
    assert summary["files"]["file_workers"] == 8
    with sqlite3.connect(tmp_path / "audit/fixture.sqlite") as db:
        assert (
            db.execute("SELECT COUNT(*) FROM physical_frames WHERE sha256 IS NOT NULL").fetchone()[
                0
            ]
            == 257
        )
