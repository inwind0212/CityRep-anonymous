from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def downloader():
    spec = importlib.util.spec_from_file_location("tessera_download", ROOT / "data_preparation/satellite/download_tessera_city.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("city", ["mumbai", "nairobi", "jakarta", "cape_town"])
def test_new_benchmark_cities_are_accepted(downloader, monkeypatch, tmp_path, city):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return {"city": kwargs["city"]}
    monkeypatch.setattr(downloader, "run_download", download)
    monkeypatch.setattr("sys.argv", ["download_tessera_city.py", "--root", str(tmp_path), "--city", city])
    downloader.main()
    assert len(calls) == 1
    assert calls[0]["city"] == city
    assert calls[0]["root"] == tmp_path


def test_all_selects_every_registered_city(downloader, monkeypatch, tmp_path):
    registry = json.loads((ROOT / "data/tasks.json").read_text())["tasks"]
    specs = registry.values() if isinstance(registry, dict) else registry
    expected = {row["city"] for row in specs}
    calls = []
    def download(**kwargs):
        calls.append(kwargs["city"])
        return {"city": kwargs["city"]}
    monkeypatch.setattr(downloader, "run_download", download)
    monkeypatch.setattr("sys.argv", ["download_tessera_city.py", "--root", str(tmp_path), "--city", "all"])
    downloader.main()
    assert set(calls) == expected
    assert len(calls) == len(expected) == 8
