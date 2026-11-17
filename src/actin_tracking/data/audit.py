"""Bounded-memory COCO import, validation and indexed audit artifacts.

Accepted source records are preserved in SQLite; rejected records produce issues.
Source files are never modified.
JSON objects and mask decoding are streamed. PNG IO uses eight workers
with batches of 32, independent of total dataset size.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO

import ijson  # type: ignore[import-untyped]
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask  # type: ignore[import-untyped]

from actin_tracking.data.diagnostics import overlap_sample, skeleton_pixels
from actin_tracking.data.schemas import exact_id


class HashReader:
    """Update the source digest as ijson consumes bytes."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_decimal)


def _decimal(value: Any) -> float:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Unsupported value: {type(value)}")


def stream_records(reader: HashReader) -> Iterator[tuple[str, Any]]:
    """Yield one top-level-array item (or info object) without buffering arrays."""
    builder: Any = None
    target = ""
    depth = 0
    for prefix, event, value in ijson.parse(reader):
        if builder is None:
            if prefix in (
                "images.item",
                "annotations.item",
                "categories.item",
                "licenses.item",
                "info",
            ):
                if event in ("start_map", "start_array"):
                    builder = ijson.ObjectBuilder()
                    target = prefix.split(".")[0]
                    depth = 1
                    builder.event(event, value)
                elif event not in ("map_key", "end_map", "end_array"):
                    yield prefix.split(".")[0], value
        else:
            builder.event(event, value)
            if event in ("start_map", "start_array"):
                depth += 1
            elif event in ("end_map", "end_array"):
                depth -= 1
            if depth == 0:
                yield target, builder.value
                builder = None


def _id(value: Any) -> str:
    return exact_id(value)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        PRAGMA journal_mode=DELETE;
        PRAGMA cache_size=-8192;
        PRAGMA temp_store=FILE;
        CREATE TABLE images(id TEXT PRIMARY KEY,file_name TEXT,sequence_id TEXT,
                            frame_index INTEGER,width INTEGER,height INTEGER,payload TEXT);
        CREATE INDEX images_sequence ON images(sequence_id,frame_index);
        CREATE INDEX images_file ON images(file_name);
        CREATE TABLE annotations(row_id INTEGER PRIMARY KEY,id TEXT,image_id TEXT,
            category_id TEXT,track_id TEXT,payload TEXT,mask_hash TEXT,selection_hash TEXT,
            area REAL,bbox_width REAL,bbox_height REAL);
        CREATE INDEX annotations_image ON annotations(image_id);
        CREATE INDEX annotations_id ON annotations(id);
        CREATE TABLE categories(id TEXT PRIMARY KEY,payload TEXT);
        CREATE TABLE issues(code TEXT,record_type TEXT,record_id TEXT,detail TEXT);
        CREATE TABLE physical_frames(file_name TEXT PRIMARY KEY,canonical_image_id TEXT,
            valid INTEGER DEFAULT 1,
            conflicted INTEGER DEFAULT 0,sha256 TEXT,bytes INTEGER);
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
    """)


def _issue(db: sqlite3.Connection, code: str, kind: str, identifier: Any, detail: Any) -> None:
    db.execute("INSERT INTO issues VALUES (?,?,?,?)", (code, kind, str(identifier), dumps(detail)))


def _image(db: sqlite3.Connection, record: dict[str, Any], synthetic: bool) -> None:
    identifier = _id(record["id"])
    filename = record.get("file_name")
    if not isinstance(filename, str) or not filename:
        raise ValueError("file_name must be a nonempty string")
    width, height = record.get("width"), record.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive integers")
    sequence = _id(record["sequence_id"]) if "sequence_id" in record else None
    frame: int | None = None
    try:
        stem = Path(filename).stem
        if synthetic:
            seq_stem, suffix = stem.rsplit("_", 1)
            frame = int(suffix)
            if sequence != seq_stem:
                _issue(db, "sequence_filename_mismatch", "image", identifier, filename)
        else:
            frame = int(stem)
    except (ValueError, AttributeError):
        _issue(db, "invalid_frame_name", "image", identifier, filename)
    if synthetic and sequence is None:
        _issue(db, "missing_sequence_id", "image", identifier, filename)
    try:
        db.execute(
            "INSERT INTO images VALUES (?,?,?,?,?,?,?)",
            (identifier, filename, sequence, frame, width, height, dumps(record)),
        )
    except sqlite3.IntegrityError:
        _issue(db, "duplicate_image_id", "image", identifier, record)
        return
    db.execute(
        "INSERT OR IGNORE INTO physical_frames(file_name,canonical_image_id) VALUES (?,?)",
        (filename, identifier),
    )


def _validate_runs(rle: dict[str, Any]) -> int:
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


def _annotation(db: sqlite3.Connection, record: dict[str, Any]) -> None:
    identifier, image, category = (_id(record[key]) for key in ("id", "image_id", "category_id"))
    track = _id(record["track_id"]) if record.get("track_id") is not None else None
    bbox: Any = record.get("bbox")
    area: Any = record.get("area")
    rle: Any = record.get("segmentation")
    valid_bbox = isinstance(bbox, list) and len(bbox) == 4 and all(_finite(v) for v in bbox)
    if not valid_bbox or any(v < 0 for v in bbox):
        _issue(db, "invalid_bbox", "annotation", identifier, bbox)
        valid_bbox = False
    if not _finite(area) or area < 0:
        _issue(db, "invalid_area", "annotation", identifier, area)
        area = None
    if area == 0:
        _issue(db, "empty_mask", "annotation", identifier, "declared area is zero")
    valid_rle = (
        isinstance(rle, dict)
        and isinstance(rle.get("size"), list)
        and len(rle["size"]) == 2
        and all(type(v) is int and v > 0 for v in rle["size"])
        and (
            isinstance(rle.get("counts"), str)
            or (
                isinstance(rle.get("counts"), list)
                and all(type(v) is int and v >= 0 for v in rle["counts"])
            )
        )
    )
    if valid_rle:
        try:
            rle_area = _validate_runs(rle)
            if rle_area == 0:
                _issue(db, "empty_rle_mask", "annotation", identifier, rle_area)
            if area is not None and abs(float(area) - rle_area) > 1e-6:
                _issue(db, "rle_area_mismatch", "annotation", identifier, rle_area)
        except (ValueError, TypeError, KeyError, IndexError):
            valid_rle = False
    if not valid_rle:
        _issue(db, "invalid_rle", "annotation", identifier, rle)
    digest = hashlib.sha256(dumps(rle).encode()).hexdigest() if valid_rle else None
    selection = hashlib.sha256(identifier.encode()).hexdigest()
    db.execute(
        "INSERT INTO annotations(id,image_id,category_id,track_id,payload,mask_hash,"
        "selection_hash,area,bbox_width,bbox_height) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            identifier,
            image,
            category,
            track,
            dumps(record),
            digest,
            selection,
            float(area) if area is not None else None,
            float(bbox[2]) if valid_bbox else None,
            float(bbox[3]) if valid_bbox else None,
        ),
    )


def _inspect_png(args: tuple[Path, str, bool]) -> dict[str, Any]:
    root, filename, sample_brightness = args
    relative = Path(filename)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts:
        return {"error": "unsafe_image_path", "detail": filename}
    try:
        # A plain filename needs one symlink check rather than resolving all parents.
        if len(relative.parts) > 1 or path.is_symlink():
            path = path.resolve()
            if not path.is_relative_to(root):
                return {"error": "unsafe_image_path", "detail": filename}
        sha = hashlib.sha256()
        size = 0
        header = b""
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                if not size:
                    header = chunk[:24]
                sha.update(chunk)
                size += len(chunk)
        result: dict[str, Any] = {"sha256": sha.hexdigest(), "bytes": size}
        if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) != 24:
            result.update(error="invalid_png", detail=filename)
        else:
            result["dimensions"] = struct.unpack(">II", header[16:24])
            if sample_brightness:
                with Image.open(path) as image:
                    result["brightness"] = float(np.asarray(image.convert("L")).mean())
        return result
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return {"error": "missing_or_unreadable_png", "detail": str(error)}


def _files(db: sqlite3.Connection, root: Path) -> dict[str, Any]:
    root = root.resolve()
    digest = hashlib.sha256()
    brightness: list[float] = []
    db.create_function("filename_hash", 1, lambda value: hashlib.sha256(value.encode()).hexdigest())
    brightness_files = {
        row[0]
        for row in db.execute(
            "SELECT file_name FROM physical_frames ORDER BY filename_hash(file_name) LIMIT 100"
        )
    }
    rows = db.execute("SELECT file_name,canonical_image_id FROM physical_frames ORDER BY file_name")
    processed = 0
    # Only 32 queued files and eight 1-MB read buffers, independent of dataset size.
    with ThreadPoolExecutor(max_workers=8) as pool:
        while batch := list(islice(rows, 32)):
            results = pool.map(
                _inspect_png, ((root, row[0], row[0] in brightness_files) for row in batch)
            )
            for (filename, identifier), result in zip(batch, results, strict=True):
                if "sha256" in result:
                    db.execute(
                        "UPDATE physical_frames SET sha256=?,bytes=? WHERE file_name=?",
                        (result["sha256"], result["bytes"], filename),
                    )
                    digest.update(dumps([filename, result["sha256"]]).encode() + b"\n")
                if "error" in result:
                    _issue(db, result["error"], "image", identifier, result["detail"])
                if "dimensions" in result:
                    width, height = result["dimensions"]
                    for image_id, w, h in db.execute(
                        "SELECT id,width,height FROM images WHERE file_name=?", (filename,)
                    ):
                        if (w, h) != (width, height):
                            _issue(
                                db, "image_dimensions_mismatch", "image", image_id, [width, height]
                            )
                if "brightness" in result:
                    brightness.append(result["brightness"])
            processed += len(batch)
            if processed % 256 == 0:
                db.commit()
            if processed % 1024 == 0:
                print(f"PNG checks: {processed}", file=sys.stderr, flush=True)
    for candidate in root.glob("*.png"):
        if (
            db.execute(
                "SELECT 1 FROM physical_frames WHERE file_name=?", (candidate.name,)
            ).fetchone()
            is None
        ):
            _issue(db, "unreferenced_png", "file", candidate.name, "No COCO image record")
    db.execute(
        "UPDATE physical_frames SET valid=0 WHERE file_name IN "
        "(SELECT i.file_name FROM images i JOIN issues x ON i.id=x.record_id WHERE x.code IN "
        "('unsafe_image_path','invalid_png','image_dimensions_mismatch',"
        "'missing_or_unreadable_png'))"
    )
    return {
        "image_content_sha256": digest.hexdigest(),
        "brightness_sample_n": len(brightness),
        "brightness_frame_mean": _numbers(brightness),
        "file_workers": 8,
        "file_queue": 32,
    }


def _numbers(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def _relations(db: sqlite3.Connection, synthetic: bool) -> None:
    queries = [
        (
            "duplicate_annotation_id",
            "SELECT id,COUNT(*) FROM annotations GROUP BY id HAVING COUNT(*)>1",
        ),
        (
            "invalid_image_ref",
            "SELECT a.id,a.image_id FROM annotations a LEFT JOIN images i ON "
            "a.image_id=i.id WHERE i.id IS NULL",
        ),
        (
            "invalid_category_ref",
            "SELECT a.id,a.category_id FROM annotations a LEFT JOIN categories c ON "
            "a.category_id=c.id WHERE c.id IS NULL",
        ),
        (
            "duplicate_mask",
            "SELECT MIN(id),COUNT(*) FROM annotations WHERE mask_hash IS NOT NULL GROUP BY "
            "image_id,mask_hash HAVING COUNT(*)>1",
        ),
        (
            "duplicate_track_in_frame",
            "SELECT MIN(id),COUNT(*) FROM annotations WHERE track_id IS NOT NULL GROUP BY "
            "image_id,track_id HAVING COUNT(*)>1",
        ),
    ]
    for code, query in queries:
        for identifier, detail in db.execute(query):
            _issue(db, code, "annotation", identifier, detail)
    for identifier, payload, width, height in db.execute(
        "SELECT a.id,a.payload,i.width,i.height FROM annotations a JOIN images i ON a.image_id=i.id"
    ):
        record = json.loads(payload)
        bbox = record.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and all(_finite(v) for v in bbox):
            x, y, w, h = bbox
            if x + w > width + 1e-6 or y + h > height + 1e-6:
                _issue(db, "bbox_out_of_bounds", "annotation", identifier, bbox)
        rle = record.get("segmentation")
        if isinstance(rle, dict) and rle.get("size") != [height, width]:
            _issue(db, "rle_dimensions_mismatch", "annotation", identifier, rle.get("size"))
    if synthetic:
        for (sequence,) in db.execute(
            "SELECT DISTINCT sequence_id FROM images WHERE sequence_id IS NOT NULL"
        ):
            frames = [
                row[0]
                for row in db.execute(
                    "SELECT frame_index FROM images WHERE sequence_id=? ORDER BY frame_index",
                    (sequence,),
                )
            ]
            if frames != list(range(30)):
                _issue(db, "incomplete_sequence", "sequence", sequence, frames)
        for (identifier,) in db.execute("SELECT id FROM annotations WHERE track_id IS NULL"):
            _issue(db, "missing_track_id", "annotation", identifier, None)
        for sequence, track, n, start, end in db.execute(
            "SELECT i.sequence_id,a.track_id,COUNT(DISTINCT i.frame_index),MIN(i.frame_index),"
            "MAX(i.frame_index) FROM annotations a JOIN images i ON a.image_id=i.id "
            "WHERE a.track_id IS NOT NULL AND i.sequence_id IS NOT NULL "
            "GROUP BY i.sequence_id,a.track_id"
        ):
            if start is not None and n != end - start + 1:
                _issue(db, "track_temporal_gap", "track", f"{sequence}/{track}", [n, start, end])


def _deduplicate(db: sqlite3.Connection) -> None:
    for (filename,) in db.execute(
        "SELECT file_name FROM images GROUP BY file_name HAVING COUNT(*)>1"
    ):
        baseline: list[str] | None = None
        for (identifier,) in db.execute(
            "SELECT id FROM images WHERE file_name=? ORDER BY rowid", (filename,)
        ):
            signatures = []
            for (payload,) in db.execute(
                "SELECT payload FROM annotations WHERE image_id=?", (identifier,)
            ):
                record = json.loads(payload)
                signatures.append(
                    dumps(
                        {
                            key: record.get(key)
                            for key in (
                                "category_id",
                                "segmentation",
                                "bbox",
                                "area",
                                "iscrowd",
                                "track_id",
                            )
                        }
                    )
                )
            signatures.sort()
            if baseline is not None and signatures != baseline:
                db.execute("UPDATE physical_frames SET conflicted=1 WHERE file_name=?", (filename,))
                _issue(db, "conflicting_real_annotations", "image", identifier, filename)
            if baseline is None:
                baseline = signatures


def _decode(db: sqlite3.Connection, sample: int, full: bool) -> dict[str, Any]:
    query = (
        "SELECT a.id,a.payload FROM annotations a "
        "JOIN images i ON i.id=a.image_id WHERE a.mask_hash IS NOT NULL "
        "ORDER BY a.selection_hash"
    )
    if not full:
        query += f" LIMIT {int(sample)}"
    checked = 0
    lengths: list[float] = []
    for identifier, payload in db.execute(query):
        record = json.loads(payload)
        try:
            rle = record["segmentation"]
            _validate_runs(rle)
            if rle["size"][0] * rle["size"][1] > 64_000_000:
                raise ValueError("Mask exceeds 64 million pixel decode limit")
            if isinstance(rle["counts"], list):
                rle = coco_mask.frPyObjects(rle, *rle["size"])
            decoded = coco_mask.decode(rle)
            if len(lengths) < sample:
                lengths.append(float(skeleton_pixels(decoded)))
            area = int(decoded.sum())
            box = coco_mask.toBbox(coco_mask.encode(np.asfortranarray(decoded))).tolist()
            if area == 0:
                _issue(db, "decoded_empty_mask", "annotation", identifier, area)
            if not _finite(record.get("area")) or abs(float(record["area"]) - area) > 1e-6:
                _issue(db, "mask_area_mismatch", "annotation", identifier, area)
            bbox = record.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    not _finite(a) or abs(float(a) - b) > 1e-6
                    for a, b in zip(bbox, box, strict=True)
                )
            ):
                _issue(db, "mask_bbox_mismatch", "annotation", identifier, box)
            checked += 1
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            _issue(db, "rle_decode_error", "annotation", identifier, str(error))
    return {
        "mode": "full" if full else "sha256_id_sample",
        "requested": sample,
        "decoded": checked,
        "skeleton_pixel_count": _numbers(lengths),
        "length_definition": "Zhang-Suen centreline pixel count; descriptive, not Euclidean length",
    }


def audit_dataset(
    name: str,
    dataset: dict[str, Any],
    output: Path,
    mask_sample: int = 100,
    full_masks: bool = False,
) -> dict[str, Any]:
    """Import immutable COCO annotations and return a JSON-serializable audit summary."""
    if mask_sample < 0:
        raise ValueError("mask_sample must be nonnegative")
    path = Path(dataset["annotations"])
    index = output / f"{name}.sqlite"
    if index.exists():
        raise FileExistsError(f"Refusing to overwrite index: {index}")
    db = sqlite3.connect(index)
    metadata: dict[str, Any] = {"licenses": [], "info": None}
    synthetic = dataset.get("source") == "synthetic"
    seen: dict[str, int] = {}
    try:
        _schema(db)
        with path.open("rb") as stream:
            reader = HashReader(stream)
            for section, record in stream_records(reader):
                seen[section] = seen.get(section, 0) + 1
                try:
                    if section == "info":
                        metadata["info"] = record
                    elif section == "licenses":
                        metadata["licenses"].append(record)
                    elif section == "images":
                        _image(db, record, synthetic)
                    elif section == "annotations":
                        _annotation(db, record)
                    elif section == "categories":
                        try:
                            db.execute(
                                "INSERT INTO categories VALUES (?,?)",
                                (_id(record["id"]), dumps(record)),
                            )
                        except sqlite3.IntegrityError:
                            _issue(
                                db, "duplicate_category_id", "category", record.get("id"), record
                            )
                except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
                    _issue(
                        db,
                        f"invalid_{section}_record",
                        section,
                        record.get("id") if isinstance(record, dict) else None,
                        str(error),
                    )
                if sum(seen.values()) % 10000 == 0:
                    db.commit()
                if sum(seen.values()) % 250000 == 0:
                    print(
                        f"{name}: imported {seen.get('annotations', 0)} annotations",
                        file=sys.stderr,
                        flush=True,
                    )
            annotation_hash = reader.digest.hexdigest()
        for section in ("images", "annotations", "categories"):
            if not seen.get(section):
                _issue(db, "empty_or_missing_section", "dataset", name, section)
        db.executescript("""
            CREATE INDEX annotation_dimensions ON annotations(area,bbox_width,bbox_height);
            CREATE INDEX annotation_masks ON annotations(image_id,mask_hash,id);
            CREATE INDEX annotation_tracks ON annotations(image_id,track_id,id);
            CREATE INDEX annotation_selection ON annotations(selection_hash);
        """)
        return finalize_index(
            db, name, dataset, index, annotation_hash, metadata, seen, mask_sample, full_masks
        )
    except ijson.JSONError as error:
        raise ValueError(f"Invalid source JSON: {error}") from error
    finally:
        db.close()


def finalize_index(
    db: sqlite3.Connection,
    name: str,
    dataset: dict[str, Any],
    index: Path,
    annotation_hash: str,
    metadata: dict[str, Any],
    seen: dict[str, int],
    mask_sample: int,
    full_masks: bool,
) -> dict[str, Any]:
    """Validate imported SQLite objects and produce bounded-size statistics."""
    path = Path(dataset["annotations"])
    synthetic = dataset.get("source") == "synthetic"
    db.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('import',?)",
        (dumps({"annotation_sha256": annotation_hash, "metadata": metadata, "seen": seen}),),
    )
    db.commit()
    print(f"{name}: checking references and identities", file=sys.stderr, flush=True)
    _relations(db, synthetic)
    if not synthetic:
        _deduplicate(db)
    print(f"{name}: checksumming PNG files", file=sys.stderr, flush=True)
    file_stats = _files(db, Path(dataset["images"]))
    print(f"{name}: decoding mask sample", file=sys.stderr, flush=True)
    mask_stats = _decode(db, mask_sample, full_masks)
    counts = {
        table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("images", "annotations", "physical_frames", "categories")
    }
    counts["sequences"] = db.execute("SELECT COUNT(DISTINCT sequence_id) FROM images").fetchone()[0]
    counts["conflicted_physical_frames"] = db.execute(
        "SELECT COUNT(*) FROM physical_frames WHERE conflicted=1"
    ).fetchone()[0]
    counts["evaluation_frames"] = db.execute(
        "SELECT COUNT(*) FROM physical_frames WHERE conflicted=0 AND valid=1 AND sha256 IS NOT NULL"
    ).fetchone()[0]
    counts["image_records"] = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    counts["unique_frames"] = db.execute("SELECT COUNT(*) FROM physical_frames").fetchone()[0]
    counts["annotations"] = db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    counts["sequences"] = db.execute("SELECT COUNT(DISTINCT sequence_id) FROM images").fetchone()[0]
    issues = dict(db.execute("SELECT code,COUNT(*) FROM issues GROUP BY code"))
    statistics: dict[str, Any] = {
        "mask_overlap_sample": overlap_sample(db) if mask_sample else None
    }
    for column in ("area", "bbox_width", "bbox_height"):
        n, lo, hi, mean = db.execute(
            f"SELECT COUNT({column}),MIN({column}),MAX({column}),AVG({column}) FROM annotations"
        ).fetchone()
        statistics[column] = {"n": n, "min": lo, "max": hi, "mean": mean}
    statistics["instances_per_frame"] = dict(
        zip(
            ("min", "max", "mean"),
            db.execute(
                "SELECT MIN(n),MAX(n),AVG(n) FROM (SELECT COUNT(a.row_id) n FROM "
                "images i LEFT JOIN annotations a ON a.image_id=i.id GROUP BY i.id)"
            ).fetchone(),
            strict=True,
        )
    )
    statistics["instances_per_frame_histogram"] = [
        list(row)
        for row in db.execute(
            "SELECT n,COUNT(*) FROM (SELECT COUNT(a.row_id) n FROM images i "
            "LEFT JOIN annotations a ON a.image_id=i.id GROUP BY i.id) GROUP BY n ORDER BY n"
        )
    ]
    statistics["density_instances_per_megapixel"] = dict(
        zip(
            ("min", "max", "mean"),
            db.execute(
                "SELECT MIN(d),MAX(d),AVG(d) FROM (SELECT COUNT(a.row_id)*1000000.0/"
                "(i.width*i.height) d FROM images i LEFT JOIN annotations a ON "
                "a.image_id=i.id GROUP BY i.id)"
            ).fetchone(),
            strict=True,
        )
    )
    # Fixed bins bound report size independently of annotation count.
    boundaries = [0, 1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 1048576]
    for column in ("area", "bbox_width", "bbox_height"):
        statistics[column]["histogram"] = [
            {
                "lower": low,
                "upper": high,
                "count": db.execute(
                    f"SELECT COUNT(*) FROM annotations WHERE {column}>=? AND {column}<?",
                    (low, high),
                ).fetchone()[0],
            }
            for low, high in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
        statistics[column]["overflow"] = db.execute(
            f"SELECT COUNT(*) FROM annotations WHERE {column}>=?", (boundaries[-1],)
        ).fetchone()[0]
    statistics["track_lengths"] = [
        list(row)
        for row in db.execute(
            "SELECT n,COUNT(*) FROM (SELECT COUNT(DISTINCT i.frame_index) n FROM "
            "annotations a JOIN images i ON a.image_id=i.id WHERE a.track_id IS NOT "
            "NULL GROUP BY i.sequence_id,a.track_id) GROUP BY n ORDER BY n"
        )
    ]
    summary = {
        "name": name,
        "source": dataset.get("source"),
        "index_path": str(index),
        "annotation_path": str(path),
        "annotation_sha256": annotation_hash,
        "annotation_bytes": path.stat().st_size,
        "counts": counts,
        "source_records_seen": seen,
        "issue_counts": issues,
        "metadata": metadata,
        "mask_validation": mask_stats,
        "statistics": statistics,
        "statistics_population": "source image records; real windows repeat physical frames",
        "files": file_stats,
        "calibration": {"fps": None, "pixel_size_um": None, "provenance": None},
        "temporal_continuity_confirmed": False,
        "limitations": [
            "Sampled skeleton pixel counts and mask overlap pairs are descriptive proxies; "
            "physical length and centreline crossing measurements require later stages.",
            "Source licenses are recorded, not externally verified.",
            "Real canonical records use first source occurrence; conflicting physical "
            "frames are excluded.",
            "Calibration/temporal continuity remain unknown unless independently verified.",
            "Duplicate masks are checked by exact RLE representation; decode checks "
            "use the reported sample.",
        ],
    }
    db.execute("INSERT INTO metadata VALUES ('summary',?)", (dumps(summary),))
    db.commit()
    return summary
