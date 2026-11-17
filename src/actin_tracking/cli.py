"""Reproducible dataset auditing and smoke runs."""

import argparse
import hashlib
import json
import platform
import random
import sqlite3
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from actin_tracking import __version__
from actin_tracking.data.schemas import SCHEMA_VERSION


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_config(path: Path, data_root: Path | None) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema_version")
    root = data_root if data_root is not None else path.parent / config["root"]
    config["root"] = str(root.resolve())
    expected = {"synthetic_train", "synthetic_val", "real_eval"}
    if set(config["datasets"]) != expected:
        raise ValueError(f"Expected dataset names: {sorted(expected)}")
    for dataset in config["datasets"].values():
        for key in ("images", "annotations", "alternative_annotations"):
            if key not in dataset:
                continue
            relative = Path(dataset[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{key} must be a path within the dataset root")
            dataset[key] = str((root / relative).resolve())
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="Record environment/config; no dataset read")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--data-root", type=Path)
    smoke.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    smoke.add_argument("--seed", type=int, default=20261117)
    smoke.add_argument("--dry-run", action="store_true")
    audit = commands.add_parser(
        "audit", aliases=["index"], help="Stream COCO into validated SQLite indexes"
    )
    audit.add_argument("--config", "--dataset", dest="config", type=Path, required=True)
    audit.add_argument("--data-root", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument(
        "--subset", choices=["synthetic_train", "synthetic_val", "real_eval"], action="append"
    )
    audit.add_argument("--mask-sample", type=int, default=100)
    audit.add_argument("--full-masks", action="store_true")
    audit.add_argument("--seed", type=int, default=20261117)
    audit.add_argument("--dry-run", action="store_true")
    preview = commands.add_parser("visualize", help="Inspect audited GT frames and sequences")
    preview.add_argument("--config", type=Path, required=True)
    preview.add_argument("--data-root", type=Path)
    preview.add_argument("--index-dir", type=Path, default=Path("outputs/stage1_audit"))
    preview.add_argument(
        "--dataset", choices=["synthetic_train", "synthetic_val", "real_eval"], required=True
    )
    selector = preview.add_mutually_exclusive_group()
    selector.add_argument("--image-id")
    selector.add_argument("--sequence-id")
    selector.add_argument("--window-id")
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--limit", type=int, default=3)
    preview.add_argument("--max-labels", type=int, default=20)
    preview.add_argument(
        "--layers",
        nargs="*",
        default=["masks", "boxes", "tracks"],
        choices=["masks", "boxes", "skeleton", "tracks", "trajectories"],
    )
    preview.add_argument("--resize", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    preview.add_argument("--pad-to", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    preview.add_argument("--crop-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"))
    preview.add_argument("--hflip", action="store_true")
    preview.add_argument("--vflip", action="store_true")
    preview.add_argument("--brightness", nargs=2, type=float, default=[1.0, 1.0])
    preview.add_argument("--contrast", nargs=2, type=float, default=[1.0, 1.0])
    preview.add_argument("--noise-std", type=float, default=0.0)
    preview.add_argument("--seed", type=int, default=20261117)
    preview.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args.config, args.data_root)
        snapshot = {"dataset": config, "seed": args.seed, "command": args.command}
        if args.command in ("audit", "index"):
            snapshot["audit_options"] = {
                "subsets": args.subset,
                "mask_sample": args.mask_sample,
                "full_masks": args.full_masks,
            }
        if args.command == "visualize":
            from actin_tracking.data.augmentations import Augmentations

            if args.limit <= 0 or args.max_labels < 0:
                raise ValueError("--limit must be positive and --max-labels nonnegative")
            augmentation_options = {
                "seed": args.seed,
                "resize": tuple(args.resize) if args.resize else None,
                "pad_to": tuple(args.pad_to) if args.pad_to else None,
                "crop_size": tuple(args.crop_size) if args.crop_size else None,
                "horizontal_flip_probability": float(args.hflip),
                "vertical_flip_probability": float(args.vflip),
                "brightness": tuple(args.brightness),
                "contrast": tuple(args.contrast),
                "noise_std": args.noise_std,
            }
            augmentations = Augmentations(**augmentation_options)
            snapshot["preview"] = {
                "split": args.dataset,
                "index_dir": str(args.index_dir.resolve()),
                "layers": args.layers,
                "max_labels": args.max_labels,
                "augmentations": augmentation_options,
            }
            selection = {
                "image_id": args.image_id,
                "sequence_id": args.sequence_id,
                "window_id": args.window_id,
                "seed": args.seed,
                "limit": args.limit,
            }
            snapshot["preview"]["selection"] = selection
        payload = json.dumps(snapshot, sort_keys=True, indent=2) + "\n"
        if args.dry_run:
            print(payload, end="")
            return 0
        if args.command == "visualize":
            from actin_tracking.data.dataset import FilamentDataset
            from actin_tracking.visualization.inspector import visualize

            snapshot["git_commit"] = git_output("rev-parse", "HEAD")
            snapshot["git_status"] = git_output("status", "--porcelain")
            with FilamentDataset.from_config(
                config, args.index_dir, args.dataset, augmentations
            ) as dataset:
                visualize(
                    dataset,
                    args.output,
                    layers=args.layers,
                    selection=selection,
                    config=snapshot,
                    max_labels=args.max_labels,
                )
            print(f"GT preview: {args.output / 'manifest.json'}")
            return 0
        if args.command in ("audit", "index"):
            from actin_tracking.data.reporting import run_audit

            if args.mask_sample < 0:
                raise ValueError("--mask-sample must be nonnegative")
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "config.resolved.json").write_text(payload, encoding="utf-8")
            result = run_audit(
                config,
                args.output,
                args.subset,
                args.mask_sample,
                args.full_masks,
                args.seed,
                git_output("rev-parse", "HEAD"),
                git_output("status", "--porcelain"),
            )
            print(f"Audit report: {args.output / 'audit_report.md'}")
            return 1 if result["issue_count"] else 0
        random.seed(args.seed)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "package_version": __version__,
            "created_utc": datetime.now(UTC).isoformat(),
            "seed": args.seed,
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status": git_output("status", "--porcelain"),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "config_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "dataset_hashes": None,
            "note": "Smoke only; no data read, audit, inference or evaluation performed.",
        }
        # Refuse to overwrite a previous run.
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.resolved.json").write_text(payload, encoding="utf-8")
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError) as error:
        parser.exit(2, f"actin: {error}\n")
    print(f"Smoke manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
