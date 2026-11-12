from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from urban_benchmark.cli import _write_main_table

SEEDS = [42, 24, 7, 0, 100]


def _city_row(root, city, values, *, task="population", metric="R2", seeds=SEEDS):
    task_id = f"{city}.{task}.2024"
    run = root / "my_model" / city / task_id / "block10_5seed_mlp1024"
    run.mkdir(parents=True)
    (run / "metrics_per_seed.json").write_text(json.dumps({
        "seeds": [{"seed": seed, metric: value} for seed, value in zip(seeds, values)],
    }))
    return {"model": "my_model", "model_label": "My embedding", "city": city,
            "task": task, "task_id": task_id, "protocol": "block10_5seed_mlp1024",
            "save_dir": str(run), metric: float(np.mean(values))}


def test_paper_table_averages_cities_before_seed_std_and_keeps_user_model(tmp_path):
    summary = pd.DataFrame([
        _city_row(tmp_path, "london", [0.2, 0.4, 0.6, 0.8, 1.0]),
        _city_row(tmp_path, "sydney", [0.8, 0.6, 0.4, 0.2, 0.0]),
    ])
    _write_main_table(summary, tmp_path)
    table = pd.read_csv(tmp_path / "main_table_avg_seedstd_paper.csv")
    assert table["model"].tolist() == ["my_model"]
    assert table.loc[0, "POP Avg (R2)"] == 0.5
    assert table.loc[0, "POP S-Std."] == 0.0
    assert "POP C-Std." not in table
    long = pd.read_csv(tmp_path / "paper_tables/main_table_avg_seedstd_long.csv")
    assert long.loc[0, "n_cities"] == 2
    assert long.loc[0, "n_seeds"] == 5
    per_seed = pd.read_csv(tmp_path / "paper_tables/task_seed_city_averages.csv")
    np.testing.assert_allclose(per_seed["score"], 0.5)


def test_seed_std_uses_population_formula_and_relocated_run_paths(tmp_path):
    rows = [
        _city_row(tmp_path, "london", [0.0, 0.1, 0.2, 0.3, 0.4]),
        _city_row(tmp_path, "sydney", [0.2, 0.3, 0.4, 0.5, 0.6]),
    ]
    for row in rows:
        row["save_dir"] = str(tmp_path / "old_location")
    _write_main_table(pd.DataFrame(rows), tmp_path)
    long = pd.read_csv(tmp_path / "paper_tables/main_table_avg_seedstd_long.csv")
    assert long.loc[0, "avg"] == pytest.approx(0.3)
    assert long.loc[0, "s_std"] == pytest.approx(np.sqrt(0.02))


def test_age_seed_statistics_exclude_cities_outside_paper_subset(tmp_path):
    summary = pd.DataFrame([
        _city_row(tmp_path, "london", [0.1, 0.2, 0.3, 0.4, 0.5], task="age_distribution", metric="KL"),
        _city_row(tmp_path, "mumbai", [10.0] * 5, task="age_distribution", metric="KL"),
    ])
    _write_main_table(summary, tmp_path)
    long = pd.read_csv(tmp_path / "paper_tables/main_table_avg_seedstd_long.csv")
    assert long.loc[0, "n_cities"] == 1
    assert long.loc[0, "avg"] == pytest.approx(0.3)
    assert long.loc[0, "s_std"] == pytest.approx(np.sqrt(0.02))


def test_missing_seed_metrics_do_not_fabricate_std_from_cities(tmp_path, capsys):
    summary = pd.DataFrame([
        {"model": "my_model", "model_label": "My embedding", "city": "london", "task": "population", "R2": 0.4},
        {"model": "my_model", "model_label": "My embedding", "city": "sydney", "task": "population", "R2": 0.6},
    ])
    _write_main_table(summary, tmp_path)
    table = pd.read_csv(tmp_path / "main_table_avg_seedstd_paper.csv")
    assert table.loc[0, "POP Avg (R2)"] == 0.5
    assert pd.isna(table.loc[0, "POP S-Std."])
    assert "S-Std. left blank" in capsys.readouterr().out


def test_unmatched_city_seed_sets_do_not_change_city_mix_between_seeds(tmp_path, capsys):
    summary = pd.DataFrame([
        _city_row(tmp_path, "london", [0.1] * 5),
        _city_row(tmp_path, "sydney", [0.9] * 4, seeds=SEEDS[:4]),
    ])
    _write_main_table(summary, tmp_path)
    table = pd.read_csv(tmp_path / "main_table_avg_seedstd_paper.csv")
    assert pd.isna(table.loc[0, "POP S-Std."])
    assert "S-Std. left blank" in capsys.readouterr().out


def test_custom_task_does_not_require_a_paper_table(tmp_path):
    summary = pd.DataFrame([
        {"model": "my_model", "model_label": "My embedding", "city": "demo", "task": "custom_task", "R2": 0.5},
    ])
    _write_main_table(summary, tmp_path)
    assert not (tmp_path / "main_table_avg_seedstd_paper.csv").exists()
