"""Run dataset audits, check split overlap, and write reviewable artifacts."""

import hashlib
import importlib.metadata
import json
import platform
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from actin_tracking.data.audit import audit_dataset


def run_audit(
    config: dict[str, Any],
    output: Path,
    subsets: list[str] | None,
    mask_sample: int,
    full_masks: bool,
    seed: int,
    commit: str | None,
    status: str | None,
) -> dict[str, Any]:
    started = time.monotonic()
    code_hashes = {
        str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(Path(__file__).parents[1].rglob("*.py"))
    }
    summaries = []
    for name in subsets or list(config["datasets"]):
        print(f"Auditing {name} ...", flush=True)
        summaries.append(
            audit_dataset(name, config["datasets"][name], output, mask_sample, full_masks)
        )
    real_comparison: dict[str, Any] | None = None
    if "real_eval" in (subsets or list(config["datasets"])):
        real = config["datasets"]["real_eval"]
        if "alternative_annotations" in real:
            alternate = {**real, "annotations": real["alternative_annotations"]}
            print("Auditing real_alternative ...", flush=True)
            summaries.append(
                audit_dataset("real_alternative", alternate, output, mask_sample, full_masks)
            )
            real_comparison = compare_real_sources(output)
    overlaps: dict[str, int] = {}
    if {s["name"] for s in summaries} >= {"synthetic_train", "synthetic_val"}:
        with sqlite3.connect(output / "synthetic_train.sqlite") as db:
            db.execute("ATTACH DATABASE ? AS val", (str(output / "synthetic_val.sqlite"),))
            for name, query in {
                "sequence_ids": "SELECT COUNT(*) FROM (SELECT sequence_id FROM images "
                "INTERSECT SELECT sequence_id FROM val.images)",
                "annotation_ids": "SELECT COUNT(*) FROM (SELECT id FROM annotations "
                "INTERSECT SELECT id FROM val.annotations)",
                "image_hashes": "SELECT COUNT(*) FROM (SELECT sha256 FROM physical_frames "
                "WHERE sha256 IS NOT NULL INTERSECT SELECT sha256 FROM val.physical_frames "
                "WHERE sha256 IS NOT NULL)",
            }.items():
                overlaps[name] = db.execute(query).fetchone()[0]
    result = {
        "schema_version": "1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset_version": Path(config["root"]).name,
        "git_commit": commit,
        "git_status": status,
        "code_sha256": code_hashes,
        "seed": seed,
        "datasets": summaries,
        "split_overlap": overlaps,
        "real_annotation_comparison": real_comparison,
        "issue_count": sum(sum(s["issue_counts"].values()) for s in summaries)
        + sum(overlaps.values())
        + len((real_comparison or {}).get("conflicting_frames", [])),
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_kib": peak_rss_kib(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                package: importlib.metadata.version(package)
                for package in ("ijson", "pycocotools", "numpy", "pillow")
            },
        },
        "config_sha256": hashlib.sha256((output / "config.resolved.json").read_bytes()).hexdigest(),
    }
    (output / "dataset_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Dataset audit",
        "",
        f"Total issues: {result['issue_count']}",
        "",
        "Sources are unchanged. Every issue is retained in each SQLite `issues` table.",
        "Geometry checks decode the reported deterministic sample; use --full-masks",
        "for exhaustive decoding. RLE structure is checked for every annotation.",
        "",
    ]
    for summary in summaries:
        lines += [f"## {summary['name']}", "", "```json", json.dumps(summary, indent=2), "```", ""]
    lines += [
        "## Split overlap",
        "",
        json.dumps(overlaps),
        "",
        "## Real source comparison",
        "",
        json.dumps(real_comparison),
        "",
    ]
    (output / "audit_report.md").write_text("\n".join(lines))
    # SVG histograms use the fixed statistical bins and require no plotting dependency.
    for summary in summaries:
        hist = summary["statistics"]["area"]["histogram"]
        maximum = max((b["count"] for b in hist), default=1) or 1
        bars = []
        for i, bucket in enumerate(hist):
            height = 160 * bucket["count"] / maximum
            bars.append(
                f'<rect x="{40 + i * 48}" y="{190 - height}" width="36" '
                f'height="{height}" fill="steelblue"/>'
                f'<text x="{40 + i * 48}" y="210" font-size="10">'
                f"{bucket['lower']}</text>"
            )
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="240">'
        svg += '<text x="20" y="20">Instance area distribution (px²), lower bin bounds</text>'
        (output / f"{summary['name']}_areas.svg").write_text(svg + "".join(bars) + "</svg>")
    return result


def compare_real_sources(output: Path) -> dict[str, Any]:
    """Compare per-frame GT multisets ignoring source-local object/image IDs."""
    from actin_tracking.data.indexing import DatasetIndex

    missing: list[str] = []
    conflicts: list[str] = []
    shared = 0
    with (
        DatasetIndex(output / "real_eval.sqlite") as primary,
        DatasetIndex(output / "real_alternative.sqlite") as alternate,
    ):
        for frame in primary.iter_unique_frames():
            filename = frame["image"]["file_name"]
            row = alternate.connection.execute(
                "SELECT canonical_image_id FROM physical_frames WHERE file_name=?",
                (filename,),
            ).fetchone()
            if row is None:
                missing.append(filename)
                continue
            other = alternate.get_frame(row[0])
            shared += 1

            def signature(record: dict[str, Any]) -> list[str]:
                fields = ("segmentation", "bbox", "area", "category_id", "iscrowd", "track_id")
                return sorted(
                    json.dumps({k: a.get(k) for k in fields}, sort_keys=True)
                    for a in record["annotations"]
                )

            if signature(frame) != signature(other):
                conflicts.append(filename)
    return {
        "shared_frames": shared,
        "missing_in_alternative": missing,
        "conflicting_frames": conflicts,
        "independent_dataset": False,
        "policy": "Primary source retained; alternate never adds evaluation frames.",
    }


def peak_rss_kib() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value // 1024 if platform.system() == "Darwin" else value)
