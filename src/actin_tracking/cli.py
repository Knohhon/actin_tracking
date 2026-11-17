"""Stage 0 CLI: resolve configuration and record a reproducible smoke run."""

import argparse
import hashlib
import json
import platform
import random
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
        for key in ("images", "annotations"):
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
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args.config, args.data_root)
        snapshot = {"dataset": config, "seed": args.seed, "command": args.command}
        payload = json.dumps(snapshot, sort_keys=True, indent=2) + "\n"
        if args.dry_run:
            print(payload, end="")
            return 0
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
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"actin: {error}\n")
    print(f"Smoke manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
