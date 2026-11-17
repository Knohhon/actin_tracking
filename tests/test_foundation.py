import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from actin_tracking.cli import main
from actin_tracking.data.schemas import FilamentInstance, SequenceMetadata

CONFIG = Path(__file__).resolve().parents[1] / "configs/datasets/v0.2.1.toml"


def test_large_ids_survive_json() -> None:
    identifier = "30909304266160180"
    instance = FilamentInstance("sequence", 0, identifier, (0, 0, 1, 1), track_id=identifier)
    restored = json.loads(json.dumps(asdict(instance)))
    assert restored["instance_id"] == identifier
    assert restored["track_id"] == identifier
    with pytest.raises(ValueError):
        SequenceMetadata("sequence", "real", fps=30)


def test_smoke_manifest_and_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "run"
    args = ["smoke", "--config", str(CONFIG), "--output", str(output), "--seed", "42"]
    assert main([*args, "--dry-run"]) == 0
    assert not output.exists()
    assert main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["seed"] == 42
    assert "git_commit" in manifest
    assert (
        manifest["config_sha256"]
        == hashlib.sha256((output / "config.resolved.json").read_bytes()).hexdigest()
    )
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_missing_config_is_actionable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["smoke", "--config", str(tmp_path / "missing.toml")])
    assert error.value.code == 2
    assert "actin:" in capsys.readouterr().err


def test_integer_identifiers_and_invalid_numeric_ids() -> None:
    from actin_tracking.data.schemas import exact_id

    assert exact_id(30909304266160180) == "30909304266160180"
    for invalid in (True, 30909304266160180.0, ""):
        with pytest.raises(ValueError):
            exact_id(invalid)


def test_config_paths_do_not_depend_on_working_directory(tmp_path: Path, monkeypatch) -> None:
    from actin_tracking.cli import resolve_config

    expected = resolve_config(CONFIG, None)
    monkeypatch.chdir(tmp_path)
    assert resolve_config(CONFIG, None) == expected
    assert resolve_config(CONFIG, Path("local-data"))["root"] == str(tmp_path / "local-data")
