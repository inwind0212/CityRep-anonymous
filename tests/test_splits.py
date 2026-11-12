from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from scipy.spatial import cKDTree

from urban_benchmark.splits import _fixed_block_ids, _metric_xy, make_splits
from urban_benchmark.tasks import Task

SEEDS = [42, 24, 7, 0, 100]


def _toy_task(n: int = 4000, seed: int = 0) -> Task:
    """Points over a ~30 km x 25 km area around London (EPSG:4326)."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-0.35, 0.10, n)
    lat = rng.uniform(51.35, 51.58, n)
    samples = gpd.GeoDataFrame(
        {"sample_id": [f"s{i}" for i in range(n)]},
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )
    y = rng.normal(size=(n, 1)).astype(np.float32)
    return Task("toy.regression", "regression", samples, y, ["label"], {})


def _assert_disjoint_cover(split, n: int, *, full_cover: bool) -> None:
    tr, va, te = (set(split.train_idx.tolist()), set(split.val_idx.tolist()), set(split.test_idx.tolist()))
    assert not (tr & va) and not (tr & te) and not (va & te)
    if full_cover:
        assert len(tr) + len(va) + len(te) == n


def test_block10_unchanged_and_disjoint():
    task = _toy_task()
    cfg = {"method": "spatial_block", "n_blocks_x": 10, "n_blocks_y": 10, "test_ratio": 0.2, "val_ratio_on_train": 0.1, "seeds": SEEDS}
    for split in make_splits(task, cfg):
        _assert_disjoint_cover(split, task.n_samples, full_cover=True)


@pytest.mark.parametrize("buffer_m", [1000.0, 2000.0])
def test_buffer_split_distance_and_paired_test_set(buffer_m):
    task = _toy_task()
    base_cfg = {"method": "spatial_block", "n_blocks_x": 10, "n_blocks_y": 10, "test_ratio": 0.2, "val_ratio_on_train": 0.1, "seeds": SEEDS}
    buf_cfg = {**base_cfg, "method": "spatial_block_buffer", "buffer_m": buffer_m}
    xy = _metric_xy(task)
    for base, buf in zip(make_splits(task, base_cfg), make_splits(task, buf_cfg)):
        _assert_disjoint_cover(buf, task.n_samples, full_cover=False)
        # Paired design: identical test set; train/val are subsets of the unbuffered ones.
        assert np.array_equal(np.sort(base.test_idx), np.sort(buf.test_idx))
        assert set(buf.train_idx.tolist()) <= set(base.train_idx.tolist())
        assert set(buf.val_idx.tolist()) <= set(base.val_idx.tolist())
        tree = cKDTree(xy[buf.test_idx])
        for idx in (buf.train_idx, buf.val_idx):
            if len(idx):
                d, _ = tree.query(xy[idx], k=1)
                assert d.min() > buffer_m
        # Units removed by the buffer must actually lie within the buffer.
        removed = np.setdiff1d(base.train_idx, buf.train_idx)
        if len(removed):
            d, _ = tree.query(xy[removed], k=1)
            assert d.max() <= buffer_m
        assert buf.meta["n_train_removed"] == len(removed)


def test_fixed_block_size_and_block_disjointness():
    task = _toy_task()
    cfg = {"method": "fixed_block", "block_size_m": 2000.0, "test_ratio": 0.2, "val_ratio_on_train": 0.1, "seeds": SEEDS}
    xy = _metric_xy(task)
    assert "UTM" in task.meta["_metric_crs"] or "326" in task.meta["_metric_crs"]
    block_ids = _fixed_block_ids(xy, 2000.0)
    # Every block spans at most 2 km in each metric axis.
    for b in np.unique(block_ids):
        pts = xy[block_ids == b]
        assert np.ptp(pts[:, 0]) <= 2000.0 and np.ptp(pts[:, 1]) <= 2000.0
    for split in make_splits(task, cfg):
        _assert_disjoint_cover(split, task.n_samples, full_cover=True)
        parts = [set(block_ids[i].tolist()) for i in (split.train_idx, split.val_idx, split.test_idx)]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
        assert split.meta["block_size_m"] == 2000.0


def test_projected_crs_is_used_directly():
    task = _toy_task()
    task.samples = task.samples.to_crs("EPSG:27700")
    xy = _metric_xy(task)
    assert task.meta["_metric_crs"].endswith("27700")
    assert np.ptp(xy[:, 0]) > 20_000  # metres, not degrees
