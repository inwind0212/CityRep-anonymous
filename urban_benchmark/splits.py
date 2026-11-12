from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .tasks import Task


@dataclass
class Split:
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict


BUFFER_METHODS = {"spatial_block_buffer", "block_buffer", "buffered_block"}
FIXED_BLOCK_METHODS = {"fixed_block", "fixed_size_block", "metric_block"}


def make_splits(task: Task, split_cfg: dict) -> list[Split]:
    seeds = [int(s) for s in split_cfg.get("seeds", [42])]
    method = str(split_cfg.get("method", "spatial_block")).lower()
    # Metric (UTM) coordinates are split-independent; compute them once per task.
    metric_xy = _metric_xy(task) if method in BUFFER_METHODS | FIXED_BLOCK_METHODS else None
    return [_make_split(task, split_cfg, seed, method, metric_xy) for seed in seeds]


def _make_split(task: Task, cfg: dict, seed: int, method: str, metric_xy: np.ndarray | None = None) -> Split:
    test_ratio = float(cfg.get("test_ratio", 0.2))
    val_ratio = float(cfg.get("val_ratio_on_train", 0.1))
    n_blocks_x = int(cfg.get("n_blocks_x", cfg.get("n_blocks", 10)))
    n_blocks_y = int(cfg.get("n_blocks_y", cfg.get("n_blocks", 10)))
    if method in {"random", "random_sample"}:
        return _random_split(task, seed, test_ratio, val_ratio, bool(cfg.get("stratify_classification", True)))
    if method in {"spatial_block", "block", "blocked"}:
        return _block_split(task, seed, test_ratio, val_ratio, n_blocks_x, n_blocks_y)
    if method in BUFFER_METHODS:
        assert metric_xy is not None
        base = _block_split(task, seed, test_ratio, val_ratio, n_blocks_x, n_blocks_y)
        return _apply_buffer(base, metric_xy, float(cfg["buffer_m"]))
    if method in FIXED_BLOCK_METHODS:
        assert metric_xy is not None
        block_ids = _fixed_block_ids(metric_xy, float(cfg["block_size_m"]))
        split = _split_by_block_ids(task, seed, test_ratio, val_ratio, block_ids)
        split.meta.update({"method": "fixed_block", "block_size_m": float(cfg["block_size_m"]), "metric_crs": task.meta.get("_metric_crs")})
        return split
    raise ValueError(f"Unsupported split method: {method}")


def _metric_xy(task: Task) -> np.ndarray:
    """Sample coordinates (unit centres) in metres, using the task's local UTM zone."""
    geom = task.samples.geometry
    if geom.crs is None:
        raise ValueError("Task CRS is required for metric spatial splits.")
    if geom.crs.is_geographic:
        utm = geom.estimate_utm_crs()
        geom = geom.to_crs(utm)
        task.meta["_metric_crs"] = utm.to_string()
    else:
        task.meta["_metric_crs"] = geom.crs.to_string()
    xy = np.column_stack([geom.x.to_numpy(dtype=np.float64), geom.y.to_numpy(dtype=np.float64)])
    if not np.isfinite(xy).all():
        raise ValueError("Non-finite metric coordinates in task samples.")
    return xy


def _fixed_block_ids(xy: np.ndarray, block_size_m: float) -> np.ndarray:
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive.")
    bx = np.floor((xy[:, 0] - xy[:, 0].min()) / block_size_m).astype(np.int64)
    by = np.floor((xy[:, 1] - xy[:, 1].min()) / block_size_m).astype(np.int64)
    return by * (int(bx.max()) + 1) + bx


def _apply_buffer(base: Split, xy: np.ndarray, buffer_m: float) -> Split:
    """Drop train/val units within ``buffer_m`` metres of any test unit.

    The test set is left unchanged, so results are paired with the unbuffered
    block split for the same seed.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(xy[base.test_idx])

    def keep_far(idx: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return idx
        dist, _ = tree.query(xy[idx], k=1, distance_upper_bound=buffer_m)
        return idx[~(dist <= buffer_m)]

    train_idx = keep_far(base.train_idx)
    val_idx = keep_far(base.val_idx)
    meta = dict(base.meta)
    meta.update(
        {
            "method": "spatial_block_buffer",
            "buffer_m": buffer_m,
            "n_train_before_buffer": int(len(base.train_idx)),
            "n_val_before_buffer": int(len(base.val_idx)),
            "n_train_removed": int(len(base.train_idx) - len(train_idx)),
            "n_val_removed": int(len(base.val_idx) - len(val_idx)),
        }
    )
    if len(train_idx) == 0:
        raise ValueError(f"Buffer of {buffer_m} m removed all training samples (seed {base.seed}).")
    return Split(base.seed, train_idx, val_idx, base.test_idx, meta)


def _random_split(task: Task, seed: int, test_ratio: float, val_ratio: float, stratify_classification: bool) -> Split:
    idx = np.arange(task.n_samples)
    stratify = task.y.astype(np.int64) if task.task_type == "classification" and stratify_classification else None
    idx_train, idx_test = train_test_split(idx, test_size=test_ratio, random_state=seed, stratify=stratify)
    stratify_train = stratify[idx_train] if stratify is not None else None
    if val_ratio > 0:
        idx_train, idx_val = train_test_split(idx_train, test_size=val_ratio, random_state=seed, stratify=stratify_train)
    else:
        idx_val = np.empty((0,), dtype=np.int64)
    return Split(seed, idx_train, idx_val, idx_test, {"method": "random", "test_ratio": test_ratio, "val_ratio_on_train": val_ratio})


def _block_ids(task: Task, n_blocks_x: int, n_blocks_y: int) -> np.ndarray:
    if {"row", "col"}.issubset(task.samples.columns):
        x = task.samples["col"].to_numpy(dtype=np.float64)
        y = task.samples["row"].to_numpy(dtype=np.float64)
    else:
        x = task.samples.geometry.x.to_numpy(dtype=np.float64)
        y = task.samples.geometry.y.to_numpy(dtype=np.float64)
    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-12)
    y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
    bx = np.floor((x - float(np.nanmin(x))) / x_span * n_blocks_x).astype(np.int64)
    by = np.floor((y - float(np.nanmin(y))) / y_span * n_blocks_y).astype(np.int64)
    bx = np.clip(bx, 0, n_blocks_x - 1)
    by = np.clip(by, 0, n_blocks_y - 1)
    return by * n_blocks_x + bx


def _block_split(task: Task, seed: int, test_ratio: float, val_ratio: float, n_blocks_x: int, n_blocks_y: int) -> Split:
    split = _split_by_block_ids(task, seed, test_ratio, val_ratio, _block_ids(task, n_blocks_x, n_blocks_y))
    split.meta.update({"n_blocks_x": n_blocks_x, "n_blocks_y": n_blocks_y})
    return split


def _split_by_block_ids(task: Task, seed: int, test_ratio: float, val_ratio: float, block_ids: np.ndarray) -> Split:
    idx = np.arange(task.n_samples)
    blocks = np.unique(block_ids)
    if len(blocks) < 3:
        raise ValueError(f"Need at least 3 non-empty spatial blocks, got {len(blocks)}")
    train_blocks, test_blocks = train_test_split(blocks, test_size=test_ratio, random_state=seed)
    if val_ratio > 0:
        train_blocks, val_blocks = train_test_split(train_blocks, test_size=val_ratio, random_state=seed)
    else:
        val_blocks = np.empty((0,), dtype=blocks.dtype)
    train_idx = idx[np.isin(block_ids, train_blocks)]
    val_idx = idx[np.isin(block_ids, val_blocks)]
    test_idx = idx[np.isin(block_ids, test_blocks)]
    return Split(
        seed,
        train_idx,
        val_idx,
        test_idx,
        {
            "method": "spatial_block",
            "n_nonempty_blocks": int(len(blocks)),
            "test_ratio": test_ratio,
            "val_ratio_on_train": val_ratio,
            "train_blocks": sorted(int(x) for x in train_blocks),
            "val_blocks": sorted(int(x) for x in val_blocks),
            "test_blocks": sorted(int(x) for x in test_blocks),
        },
    )
