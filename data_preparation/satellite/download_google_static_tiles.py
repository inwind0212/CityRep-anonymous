#!/usr/bin/env python
"""Download per-region RGB satellite images via Google Maps Static API.

One image is downloaded at the centre of each H3 cell. The tiles used in the
paper are listed in metadata/inputs/<city>/google_static_tiles_musecl.csv.gz
and can be passed with --regions. Requires a Google Maps Platform API key.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_GOOGLE_STATIC_ROOT = ROOT / "data" / "raw" / "remote_sensing" / "google_maps_static"
DEFAULT_PROCESSED_ROOT = ROOT / "data" / "processed" / "musecl"


def normalize_city_name(city: str) -> str:
    return str(city).strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True)
    parser.add_argument("--regions", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_GOOGLE_STATIC_ROOT)
    parser.add_argument("--h3-res", type=int, default=8)
    parser.add_argument("--zoom", type=int, default=17)
    parser.add_argument("--size", default="640x640")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--format", default="jpg")
    parser.add_argument("--maptype", default="satellite")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--max-regions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city = normalize_city_name(args.city)
    regions_path = args.regions or (DEFAULT_PROCESSED_ROOT / city / f"regions_h3r{args.h3_res}.parquet")
    api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("Google Maps Static API key missing. Provide --api-key or env GOOGLE_MAPS_API_KEY.")

    regions = pd.read_csv(regions_path, dtype={"h3_id": str}) if str(regions_path).endswith((".csv", ".csv.gz")) else pd.read_parquet(regions_path)
    if args.max_regions is not None:
        regions = regions.head(args.max_regions).copy()

    run_dir = args.out_root / city / f"h3r{args.h3_res}_google_static_satellite_z{args.zoom}_{args.size}_x{args.scale}"
    run_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    rows = []
    errors = []
    started = time.time()

    for row in tqdm(regions.itertuples(index=False), total=len(regions), desc="download rs", ascii=True):
        h3_id = str(row.h3_id)
        out_path = run_dir / f"{h3_id}.{args.format}"
        if out_path.exists() and not args.overwrite:
            rows.append(
                {
                    "h3_id": h3_id,
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                    "filepath": str(out_path),
                    "status": "exists",
                }
            )
            continue

        params = {
            "center": f"{float(row.lat):.7f},{float(row.lon):.7f}",
            "zoom": int(args.zoom),
            "size": str(args.size),
            "scale": int(args.scale),
            "format": str(args.format),
            "maptype": str(args.maptype),
            "key": api_key,
        }
        resp = session.get("https://maps.googleapis.com/maps/api/staticmap", params=params, timeout=60)
        content_type = str(resp.headers.get("content-type", ""))
        if resp.ok and content_type.startswith("image/"):
            out_path.write_bytes(resp.content)
            rows.append(
                {
                    "h3_id": h3_id,
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                    "filepath": str(out_path),
                    "status": "downloaded",
                }
            )
        else:
            error_row = {
                "h3_id": h3_id,
                "lat": float(row.lat),
                "lon": float(row.lon),
                "status_code": int(resp.status_code),
                "content_type": content_type,
                "body_preview": resp.text[:500] if "text" in content_type or "json" in content_type else "",
            }
            errors.append(error_row)
        if args.sleep_seconds > 0:
            time.sleep(float(args.sleep_seconds))

    manifest = pd.DataFrame(rows).sort_values("h3_id").reset_index(drop=True)
    manifest_path = run_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    if errors:
        (run_dir / "download_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    meta = {
        "city": city,
        "regions": str(regions_path),
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "n_regions_requested": int(len(regions)),
        "n_downloaded_or_existing": int(len(manifest)),
        "n_errors": int(len(errors)),
        "h3_res": int(args.h3_res),
        "zoom": int(args.zoom),
        "size": str(args.size),
        "scale": int(args.scale),
        "format": str(args.format),
        "maptype": str(args.maptype),
        "elapsed_seconds": time.time() - started,
        "source": "Google Maps Static API satellite",
    }
    (run_dir / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
