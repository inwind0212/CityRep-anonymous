#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import geopandas as gpd


CITIES = ["london", "new_york", "singapore", "sydney", "mumbai", "nairobi", "jakarta", "cape_town"]
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def city_bounds(boundary_path: Path) -> tuple[float, float, float, float]:
    gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def run_download(
    *,
    city: str,
    root: Path,
    year: int,
    dataset_version: str,
    dry_run: bool,
    skip_hash: bool,
) -> dict:
    boundary_path = root / "data" / "raw" / "boundaries" / f"{city}.geojson"
    out_dir = root / "outputs" / "tessera" / city / "raw" / str(year)
    bbox = city_bounds(boundary_path)
    cmd = [
        "geotessera",
        "download",
        "--region-file",
        str(boundary_path),
        "--year",
        str(year),
        "--dataset-version",
        dataset_version,
        "--format",
        "tiff",
        "--output",
        str(out_dir),
        "--list-files",
    ]
    if skip_hash:
        cmd.append("--skip-hash")
    if dry_run:
        cmd.append("--dry-run")

    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    tile_paths = sorted(str(p) for p in out_dir.rglob("*.tif")) + sorted(str(p) for p in out_dir.rglob("*.tiff"))
    meta = {
        "city": city,
        "year": int(year),
        "dataset_version": dataset_version,
        "boundary_path": str(boundary_path),
        "bbox": bbox,
        "output_dir": str(out_dir),
        "dry_run": bool(dry_run),
        "skip_hash": bool(skip_hash),
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "n_downloaded_tifs": len(tile_paths),
        "tile_paths": tile_paths,
        "source": "GeoTessera CLI",
        "source_url": "https://github.com/ucam-eo/geotessera",
    }
    suffix = "dry_run" if dry_run else "download"
    (out_dir / f"{city}_tessera_{year}_{suffix}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official GeoTessera embedding tiles for benchmark city boundaries.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--city", choices=CITIES + ["all"], default="all")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-hash", action="store_true")
    args = parser.parse_args()

    cities = CITIES if args.city == "all" else [args.city]
    metas = [
        run_download(
            city=city,
            root=args.root,
            year=args.year,
            dataset_version=args.dataset_version,
            dry_run=args.dry_run,
            skip_hash=not args.no_skip_hash,
        )
        for city in cities
    ]
    print(json.dumps(metas, indent=2))


if __name__ == "__main__":
    main()
