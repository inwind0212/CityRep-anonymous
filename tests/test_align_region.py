from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from urban_benchmark.align import align_embedding
from urban_benchmark.embeddings import RegionEmbedding
from urban_benchmark.tasks import Task

RES = 8


def _grid_task(tmp: Path, cell_deg: float, n: int = 6) -> Task:
    """n x n regression task on a WGS84 label raster near central London."""
    tr = from_origin(-0.12, 51.52, cell_deg, cell_deg)
    path = tmp / "labels.tif"
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:4326", transform=tr) as dst:
        dst.write(np.ones((1, n, n), dtype=np.float32))
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    xs = tr.c + tr.a * (cols + 0.5)
    ys = tr.f + tr.e * (rows + 0.5)
    samples = gpd.GeoDataFrame(
        {"sample_id": [f"s{i}" for i in range(len(rows))], "row": rows, "col": cols},
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:4326",
    )
    y = np.zeros((len(rows), 1), dtype=np.float32)
    return Task("toy.reg", "regression", samples, y, ["label"], {"labels_path": str(path)})


def _region(tmp: Path, cells: list[str], values: np.ndarray) -> RegionEmbedding:
    df = pd.DataFrame(values, columns=[f"e{i}" for i in range(values.shape[1])])
    df.insert(0, "h3_id", cells)
    path = tmp / "region.parquet"
    df.to_parquet(path, index=False)
    return RegionEmbedding(name="toy", path=str(path), region_id_col="h3_id", region_type="h3", h3_resolution=RES)


def _cells_around(task: Task) -> list[str]:
    c = h3.latlng_to_cell(51.515, -0.11, RES)
    return sorted(h3.grid_disk(c, 4))


def test_area_mean_constant_embedding_and_point_fallback(tmp_path):
    task = _grid_task(tmp_path, 0.004)  # ~450 m cells: most span several hexes
    cells = _cells_around(task)
    emb = _region(tmp_path, cells, np.tile(np.array([[1.0, 2.0, 3.0]]), (len(cells), 1)))
    a = align_embedding(task, emb, normalize=False, region_method="area_mean")
    c = align_embedding(task, emb, normalize=False, region_method="centroid")
    assert a.report["aligner"] == "h3_area_mean"
    # Mean of identical vectors equals the vector itself.
    assert np.allclose(a.X, [1.0, 2.0, 3.0], atol=1e-5)
    assert np.allclose(c.X, [1.0, 2.0, 3.0], atol=1e-5)
    assert a.report["share_units_multi_cell"] > 0


def test_area_mean_weights_match_subpoint_shares(tmp_path):
    task = _grid_task(tmp_path, 0.004)
    cells = _cells_around(task)
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(len(cells), 4))
    emb = _region(tmp_path, cells, vals)
    a = align_embedding(task, emb, normalize=False, region_method="area_mean")
    s = a.report["area_subsample_per_axis"]
    lut = dict(zip(cells, vals))
    # Brute-force reference for one unit.
    i = 7
    r, cc = task.samples.loc[i, ["row", "col"]]
    with rasterio.open(task.meta["labels_path"]) as src:
        tr = src.transform
    offs = (np.arange(s) + 0.5) / s
    ref = [lut[h3.latlng_to_cell(tr.f + tr.e * (r + dr), tr.c + tr.a * (cc + dc), RES)] for dr in offs for dc in offs]
    assert np.allclose(a.X[i], np.mean(ref, axis=0), atol=1e-5)


def test_missing_cells_are_ignored_not_zero_weighted(tmp_path):
    task = _grid_task(tmp_path, 0.004)
    cells = _cells_around(task)
    keep = cells[::2]  # drop half the cells -> holes
    emb = _region(tmp_path, keep, np.ones((len(keep), 2)))
    a = align_embedding(task, emb, normalize=False, region_method="area_mean")
    c = align_embedding(task, emb, normalize=False, region_method="centroid")
    hit = np.linalg.norm(a.X, axis=1) > 0
    assert np.allclose(a.X[hit], 1.0)  # holes excluded, not averaged in as zeros
    assert a.report["hit_rate_before_fill"] >= c.report["hit_rate_before_fill"]


def test_fine_cells_inside_one_hex_equal_centroid(tmp_path):
    task = _grid_task(tmp_path, 0.0002)  # ~20 m cells, block well inside hexes
    cells = _cells_around(task)
    emb = _region(tmp_path, cells, np.random.default_rng(1).normal(size=(len(cells), 3)))
    a = align_embedding(task, emb, normalize=False, region_method="area_mean")
    c = align_embedding(task, emb, normalize=False, region_method="centroid")
    single = np.array([len({h3.latlng_to_cell(y, x, RES)}) for x, y in zip(task.samples.geometry.x, task.samples.geometry.y)])
    assert single.all()
    same = np.isclose(a.X, c.X, atol=1e-5).all(axis=1)
    assert same.mean() > 0.8
