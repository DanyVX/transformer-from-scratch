"""Packaged YAML profiles still load outside their original source checkout."""

import shutil
from pathlib import Path

from tfs import config as config_module
from tfs.config import PROFILE_NAMES, load_profile


def test_wheel_profile_fallback(tmp_path: Path, monkeypatch) -> None:
    actual_config_dir = Path(config_module.__file__).parent / "configs"
    assert actual_config_dir.is_dir()
    simulated_package = tmp_path / "site-packages" / "tfs"
    packaged_profiles = simulated_package / "configs"
    packaged_profiles.mkdir(parents=True)
    for name in PROFILE_NAMES:
        shutil.copy2(actual_config_dir / f"{name}.yaml", packaged_profiles / f"{name}.yaml")

    monkeypatch.setattr(config_module, "__file__", str(simulated_package / "config.py"))
    for name in PROFILE_NAMES:
        config = load_profile(name)
        assert config.profile == name
        assert config.model.parameter_count() > 0
