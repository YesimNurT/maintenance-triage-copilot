"""Smoke tests: the package imports and settings load without a .env file."""

from pathlib import Path

import mtc
from mtc.config import Settings


def test_package_imports():
    assert mtc.__version__


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    s = Settings()
    assert s.random_seed == 42
    assert s.sample_dir == Path("data/sample")
    assert s.google_api_key is None
