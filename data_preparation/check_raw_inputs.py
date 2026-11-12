#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]  # repository root
RAW = ROOT / "data" / "raw"
OUT = RAW / "training_raw_manifest.json"

CITIES = [
    "london",
    "new_york",
    "singapore",
    "sydney",
    "mumbai",
    "nairobi",
    "jakarta",
    "cape_town",
]


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / 1024 / 1024, 2)


def boundary_summary(city: str) -> dict[str, object]:
    path = RAW / "boundaries" / f"{city}.geojson"
    out: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    gdf = gpd.read_file(path).to_crs(4326)
    out.update(
        {
            "geometry_type": str(gdf.geometry.union_all().geom_type),
            "bounds": [round(float(x), 6) for x in gdf.total_bounds],
        }
    )
    return out


def poi_summary(city: str) -> dict[str, object]:
    city_dir = RAW / "poi" / "foursquare" / city
    paths = sorted(city_dir.glob("fsq_os_places_*.parquet"))
    out: dict[str, object] = {
        "path": str(paths[-1]) if paths else None,
        "exists": bool(paths),
        "source": "foursquare_open_source_places",
    }
    if paths:
        out["count"] = int(len(pd.read_parquet(paths[-1], columns=["fsq_place_id"])))
        out["size_mb"] = file_size_mb(paths[-1])
    return out


def alphaearth_summary(city: str) -> dict[str, object]:
    path = RAW / "alphaearth" / city / "alphaearth_2024.tif"
    out: dict[str, object] = {"path": str(path), "exists": path.exists(), "source": "alphaearth_2024"}
    if path.exists():
        with rasterio.open(path) as src:
            out.update(
                {
                    "bands": int(src.count),
                    "width": int(src.width),
                    "height": int(src.height),
                    "crs": str(src.crs),
                    "size_mb": file_size_mb(path),
                }
            )
    return out


def mapillary_summary(city: str) -> dict[str, object]:
    city_dir = RAW / "street_view" / "mapillary" / city
    meta_path = city_dir / "metadata.parquet"
    selected_path = city_dir / "metadata_selected.parquet"
    images_dir = city_dir / "images"
    out: dict[str, object] = {
        "metadata_path": str(meta_path),
        "metadata_exists": meta_path.exists(),
        "selected_path": str(selected_path),
        "selected_exists": selected_path.exists(),
        "images_dir": str(images_dir),
        "source": "mapillary",
    }
    if meta_path.exists():
        out["metadata_count"] = int(len(pd.read_parquet(meta_path, columns=["image_id"])))
        out["metadata_size_mb"] = file_size_mb(meta_path)
    if selected_path.exists():
        out["selected_count"] = int(len(pd.read_parquet(selected_path, columns=["image_id"])))
    if images_dir.exists():
        out["downloaded_image_count"] = len(list(images_dir.glob("*.jpg")))
    return out


def main() -> None:
    manifest = {
        "_meta": {
            "root": str(ROOT),
            "purpose": "Raw training data availability for urban representation baselines.",
            "canonical_layout": {
                "boundary": "data/raw/boundaries/{city}.geojson",
                "poi": "data/raw/poi/foursquare/{city}/fsq_os_places_{snapshot}.parquet",
                "alphaearth": "data/raw/alphaearth/{city}/alphaearth_2024.tif",
                "mapillary_metadata": "data/raw/street_view/mapillary/{city}/metadata.parquet",
                "mapillary_selected": "data/raw/street_view/mapillary/{city}/metadata_selected.parquet",
                "mapillary_images": "data/raw/street_view/mapillary/{city}/images/*.jpg",
            },
        },
        "cities": {},
    }
    for city in CITIES:
        manifest["cities"][city] = {
            "boundary": boundary_summary(city),
            "poi_foursquare": poi_summary(city),
            "alphaearth": alphaearth_summary(city),
            "mapillary": mapillary_summary(city),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
