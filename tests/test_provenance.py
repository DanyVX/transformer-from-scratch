import json
from pathlib import Path

import pytest

from tfs.cli import main
from tfs.provenance import environment_metadata, write_result


def test_snapshot_contains_essential_hardware_and_commit_fields() -> None:
    snapshot = environment_metadata()
    assert isinstance(snapshot["python"], str)
    assert "cuda_available" in snapshot
    assert "memory_bytes" in snapshot
    assert "git_commit" in snapshot
    assert "internet_access" in snapshot


def test_result_writes_metadata_and_metrics_atomically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"
    write_result(path, {"loss": 0.5}, config={"seed": 123})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["metrics"] == {"loss": 0.5}
    assert saved["config"]["seed"] == 123
    assert saved["environment"]["python"]
    assert len(list(path.parent.glob("*.tmp"))) == 0


def test_nan_results_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        write_result(tmp_path / "bad.json", {"loss": float("nan")}, config={})


def test_cli_prints_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profiles"]) == 0
    assert "tiny" in capsys.readouterr().out


def test_cli_prints_model_profile(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "--profile", "tiny"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["parameter_count_measured"] is False
    assert data["profile"] == "tiny"
