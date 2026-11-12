#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import reproject, transform_bounds


DEFAULT_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = RELEASE_ROOT / "data" / "external" / "alphaearth" / "aef_index.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/crop AlphaEarth Foundations annual embeddings for a benchmark city."
    )
    parser.add_argument("--city", required=True, help="Benchmark city id, e.g. new_york.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resolution-deg",
        type=float,
        default=10.0 / 111_320.0,
        help="Output pixel size in degrees. Default is approximately 10 m at the equator.",
    )
    parser.add_argument(
        "--pad-pixels",
        type=int,
        default=0,
        help="Optional bbox padding in output pixels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output tif.",
    )
    parser.add_argument(
        "--local-tile-dir",
        type=Path,
        default=None,
        help="Optional directory containing local AEF tiles named <fid>.tiff; used before remote URLs.",
    )
    return parser.parse_args()


def _download_index(index_path: Path, index_url: str) -> None:
    if index_path.is_file():
        return
    import subprocess

    index_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-o", str(index_path), index_url],
        check=True,
    )


def _source_url(location_or_path: str) -> str:
    path = str(location_or_path)
    if path.startswith("VRT://vsis3/us-west-2.opendata.source.coop/"):
        path = path.replace("VRT://vsis3/us-west-2.opendata.source.coop/", "https://data.source.coop/", 1)
    elif path.startswith("VRT://vsis3/"):
        path = path.replace("VRT://vsis3/", "https://data.source.coop/", 1)
    elif path.startswith("s3://us-west-2.opendata.source.coop/"):
        path = path.replace("s3://us-west-2.opendata.source.coop/", "https://data.source.coop/", 1)
    elif path.startswith("/vsis3/us-west-2.opendata.source.coop/"):
        path = path.replace("/vsis3/us-west-2.opendata.source.coop/", "https://data.source.coop/", 1)
    elif path.startswith("/vsis3/"):
        path = path.replace("/vsis3/", "https://data.source.coop/", 1)
    if path.endswith(".tiff"):
        path = path[:-5] + ".vrt"
    return "/vsicurl/" + path if path.startswith("https://") else path


def _source_path(row: Any, local_tile_dir: Path | None) -> str:
    if local_tile_dir is not None:
        local_tile = local_tile_dir / f"{int(row.fid)}.tiff"
        if local_tile.is_file():
            return str(local_tile)
    return _source_url(row.location or row.path)


def _aef_int8_to_float(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float32)
    valid = values != -128
    valid_values = values[valid].astype(np.float32)
    out[valid] = np.sign(valid_values) * np.square(valid_values / 127.5)
    return out


def _clipped_window(bounds: tuple[float, float, float, float], transform: rasterio.Affine, width: int, height: int) -> Window | None:
    left, bottom, right, top = bounds
    win = from_bounds(left, bottom, right, top, transform=transform)
    col0 = max(0, int(math.floor(win.col_off)))
    row0 = max(0, int(math.floor(win.row_off)))
    col1 = min(width, int(math.ceil(win.col_off + win.width)))
    row1 = min(height, int(math.ceil(win.row_off + win.height)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def _select_tiles(index_path: Path, year: int, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    columns = [
        "fid",
        "crs",
        "path",
        "year",
        "utm_zone",
        "wgs84_west",
        "wgs84_south",
        "wgs84_east",
        "wgs84_north",
        "location",
    ]
    df = pd.read_parquet(index_path, columns=columns)
    left, bottom, right, top = bounds
    selected = df[
        (df["year"] == year)
        & (df["wgs84_east"] >= left)
        & (df["wgs84_west"] <= right)
        & (df["wgs84_north"] >= bottom)
        & (df["wgs84_south"] <= top)
    ].copy()
    return selected.sort_values(["utm_zone", "fid"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it.")

    _download_index(args.index, args.index_url)
    boundary = gpd.read_file(args.boundary).to_crs("EPSG:4326")
    west, south, east, north = (float(x) for x in boundary.total_bounds)
    pad = args.pad_pixels * args.resolution_deg
    bounds = (west - pad, south - pad, east + pad, north + pad)

    tiles = _select_tiles(args.index, args.year, bounds)
    if tiles.empty:
        raise RuntimeError(f"No AEF {args.year} tiles intersect {args.city} bounds {bounds}.")

    xres = float(args.resolution_deg)
    yres = float(args.resolution_deg)
    width = int(math.ceil((bounds[2] - bounds[0]) / xres))
    height = int(math.ceil((bounds[3] - bounds[1]) / yres))
    transform = from_origin(bounds[0], bounds[3], xres, yres)
    dst_crs = "EPSG:4326"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources: list[dict[str, Any]] = []
    open_datasets: list[rasterio.io.DatasetReader] = []
    try:
        for row in tiles.itertuples(index=False):
            url = _source_path(row, args.local_tile_dir)
            src = rasterio.open(url)
            if src.count != 64:
                raise RuntimeError(f"Expected 64 bands in {url}, found {src.count}.")
            src_bounds_4326 = transform_bounds(src.crs, dst_crs, *src.bounds, densify_pts=21)
            inter = (
                max(bounds[0], src_bounds_4326[0]),
                max(bounds[1], src_bounds_4326[1]),
                min(bounds[2], src_bounds_4326[2]),
                min(bounds[3], src_bounds_4326[3]),
            )
            win = _clipped_window(inter, transform, width, height)
            if win is None:
                src.close()
                continue
            open_datasets.append(src)
            sources.append(
                {
                    "fid": int(row.fid),
                    "utm_zone": row.utm_zone,
                    "crs": row.crs,
                    "url": url,
                    "src": src,
                    "window": win,
                }
            )

        if not sources:
            raise RuntimeError("Tile bbox selection produced no usable raster intersections.")

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 64,
            "dtype": "float32",
            "crs": dst_crs,
            "transform": transform,
            "compress": "lzw",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "YES",
            "nodata": None,
        }

        with rasterio.open(args.output, "w", **profile) as dst:
            for band_idx in range(1, 65):
                band = np.full((height, width), -128, dtype=np.int8)
                for item in sources:
                    src = item["src"]
                    win: Window = item["window"]
                    row0 = int(win.row_off)
                    row1 = int(win.row_off + win.height)
                    col0 = int(win.col_off)
                    col1 = int(win.col_off + win.width)
                    temp = np.full((int(win.height), int(win.width)), -128, dtype=np.int8)
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=temp,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=-128,
                        dst_transform=window_transform(win, transform),
                        dst_crs=dst_crs,
                        dst_nodata=-128,
                        resampling=Resampling.nearest,
                        num_threads=2,
                    )
                    valid = temp != -128
                    if np.any(valid):
                        band[row0:row1, col0:col1][valid] = temp[valid]
                dst.write(_aef_int8_to_float(band), band_idx)
                print(f"Wrote band {band_idx}/64", flush=True)
    finally:
        for src in open_datasets:
            src.close()

    meta = {
        "city": args.city,
        "year": args.year,
        "source": "Source Cooperative tge-labs/aef AlphaEarth Foundations annual COGs",
        "source_index_url": args.index_url,
        "source_index_path": str(args.index.resolve()),
        "boundary": str(args.boundary.resolve()),
        "output": str(args.output.resolve()),
        "output_crs": dst_crs,
        "output_transform": tuple(float(x) for x in transform),
        "output_width": width,
        "output_height": height,
        "output_count": 64,
        "output_dtype": "float32",
        "resolution_deg": args.resolution_deg,
        "bounds": bounds,
        "tile_count": len(sources),
        "tiles": [
            {
                "fid": item["fid"],
                "utm_zone": item["utm_zone"],
                "crs": item["crs"],
                "url": item["url"],
                "window": {
                    "row_off": int(item["window"].row_off),
                    "col_off": int(item["window"].col_off),
                    "height": int(item["window"].height),
                    "width": int(item["window"].width),
                },
            }
            for item in sources
        ],
        "value_transform": "sign(x) * (x / 127.5)^2 for int8 source values; source nodata -128 is written as zero vector.",
        "attribution": "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.",
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
