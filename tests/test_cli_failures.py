from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from urban_benchmark import cli


def _manifest(root, models, *, files_exist):
    rows = []
    for model in models:
        path = root / f"{model}.tif"
        if files_exist:
            path.write_bytes(b"fixture")
        rows.append({"model": model, "model_label": model, "city": "london",
                     "task": "population", "task_id": "london.population.2024",
                     "available": True,
                     "embedding_config": json.dumps({"type": "raster", "name": model, "path": str(path)})})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_missing_all_inputs_exits_nonzero_and_saves_failures(tmp_path):
    manifest = _manifest(tmp_path, ["missing"], files_exist=False)
    out = tmp_path / "results"
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate", "--embedding-manifest", str(manifest), "--out-root", str(out)])
    assert error.value.code == 1
    assert pd.read_csv(out / "failures.csv")["error"].tolist() == ["FileNotFoundError"]
    assert (out / "results.csv").is_file()


def test_dry_run_of_missing_inputs_only_inspects_paths(tmp_path):
    manifest = _manifest(tmp_path, ["missing"], files_exist=False)
    out = tmp_path / "results"
    assert cli.main(["evaluate", "--embedding-manifest", str(manifest), "--out-root", str(out), "--dry-run"]) is None
    assert (out / "manifest.csv").is_file()
    assert not (out / "failures.csv").exists()


def _fake_evaluator(monkeypatch, *, fail):
    called = []
    monkeypatch.setattr(cli, "load_task", lambda *a, **k: SimpleNamespace(task_type="regression", n_samples=2))
    def evaluate(task, embedding, **kwargs):
        called.append(embedding.name)
        if embedding.name == fail:
            raise RuntimeError("fixture failure")
        return SimpleNamespace(metrics={"R2": 0.5}, metrics_std={"R2": 0.01}, save_dir=str(kwargs["out_dir"]))
    monkeypatch.setattr(cli, "evaluate_task", evaluate)
    return called


def test_partial_failures_finish_batch_save_successes_and_exit_nonzero(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, ["bad", "good"], files_exist=True)
    called = _fake_evaluator(monkeypatch, fail="bad")
    out = tmp_path / "results"
    with pytest.raises(SystemExit) as error:
        cli.main(["evaluate", "--embedding-manifest", str(manifest), "--out-root", str(out)])
    assert error.value.code == 1
    assert called == ["bad", "good"]
    assert pd.read_csv(out / "results.csv")["model"].tolist() == ["good"]
    assert pd.read_csv(out / "failures.csv")["model"].tolist() == ["bad"]


def test_successful_batch_returns_normally(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path, ["good"], files_exist=True)
    called = _fake_evaluator(monkeypatch, fail=None)
    assert cli.main(["evaluate", "--embedding-manifest", str(manifest), "--out-root", str(tmp_path / "results")]) is None
    assert called == ["good"]
