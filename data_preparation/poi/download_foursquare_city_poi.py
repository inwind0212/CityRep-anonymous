#!/usr/bin/env python
from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]  # repository root
BOUNDARY_DIR = ROOT / "data" / "raw" / "boundaries"
OUT_DIR = ROOT / "data" / "raw" / "poi" / "foursquare"
SUMMARY_PATH = OUT_DIR / "summary.json"

CITIES = [
    "london",
    "new_york",
    "singapore",
    "sydney",
    "mumbai",
    "nairobi",
    "jakarta",
    "lagos",
    "cape_town",
]
SOURCE_COOP_ROOT = "https://source.coop/fused/fsq-os-places"
SOURCE_COOP_DATA_ROOT = "https://data.source.coop/fused/fsq-os-places"


def latest_release_date() -> str:
    html = requests.get(SOURCE_COOP_ROOT, timeout=60).text
    dates = sorted(set(re.findall(r'\\"path\\":\\"(\d{4}-\d{2}-\d{2})/\\"', html)))
    if not dates:
        raise RuntimeError("Could not discover Foursquare release dates from Source Cooperative.")
    return dates[-1]


def release_urls(dt: str) -> list[str]:
    html = requests.get(f"{SOURCE_COOP_ROOT}/{dt}/places", timeout=60).text
    paths = sorted(set(re.findall(rf'{re.escape(dt)}/places/\d+\.parquet', html)))
    if not paths:
        raise RuntimeError(f"Could not discover parquet files for release {dt}.")
    return [f"{SOURCE_COOP_DATA_ROOT}/{path}" for path in paths]


def query_bbox_subset(urls: list[str], bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    minx, miny, maxx, maxy = bounds
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    query = """
        SELECT *
        FROM read_parquet($urls)
        WHERE longitude BETWEEN $minx AND $maxx
          AND latitude BETWEEN $miny AND $maxy
    """
    return con.execute(
        query,
        {
            "urls": urls,
            "minx": minx,
            "miny": miny,
            "maxx": maxx,
            "maxy": maxy,
        },
    ).fetch_df()


def clip_to_boundary(df: pd.DataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    clipped = gpd.sjoin(gdf, boundary[["geometry"]], how="inner", predicate="within")
    clipped = clipped.drop(columns=["index_right"], errors="ignore")
    return clipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Foursquare Open Source Places for benchmark city boundaries."
    )
    parser.add_argument(
        "cities",
        nargs="*",
        default=CITIES,
        help="City ids to download. Defaults to all configured benchmark cities.",
    )
    parser.add_argument(
        "--release",
        default="2025-02-06",
        help="Foursquare Open Source Places release date (dt=...). CityRep uses 2025-02-06; pass 'latest' for the newest release.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    latest_dt = latest_release_date() if args.release == "latest" else args.release
    urls = release_urls(latest_dt)
    print(f"[info] using Foursquare Source Cooperative snapshot: {latest_dt}")
    print(f"[info] parquet files: {len(urls)}")

    summary: dict[str, dict[str, object]] = {
        "_meta": {
            "dataset": "Foursquare Open Source Places",
            "distribution": "Source Cooperative public mirror",
            "snapshot_date": latest_dt,
            "n_remote_parquet_files": len(urls),
        }
    }

    for city in args.cities:
        boundary_path = BOUNDARY_DIR / f"{city}.geojson"
        boundary = gpd.read_file(boundary_path).to_crs(4326)

        bbox_df = query_bbox_subset(urls, tuple(boundary.total_bounds))
        city_gdf = clip_to_boundary(bbox_df, boundary)

        city_dir = OUT_DIR / city
        city_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = city_dir / f"fsq_os_places_{latest_dt}.parquet"
        meta_path = city_dir / f"fsq_os_places_{latest_dt}.meta.json"

        city_gdf.to_parquet(parquet_path, index=False)

        meta = {
            "city": city,
            "snapshot_date": latest_dt,
            "count": int(len(city_gdf)),
            "boundary_path": str(boundary_path),
            "parquet_path": str(parquet_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        summary[city] = {
            "count": int(len(city_gdf)),
            "parquet": str(parquet_path),
        }
        print(f"[saved] {city}: {len(city_gdf)} POIs")

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[saved] summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
