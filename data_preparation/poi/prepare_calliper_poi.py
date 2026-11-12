#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


CITY_PROJECTED_CRS = {
    "london": "EPSG:27700",
    "singapore": "EPSG:3414",
    "new_york": "EPSG:32618",
    "sydney": "EPSG:32756",
    "mumbai": "EPSG:32643",
    "nairobi": "EPSG:32737",
    "jakarta": "EPSG:32748",
    "cape_town": "EPSG:32734",
    "lagos": "EPSG:32631",
}


def prepare_city(city: str, poi_path: Path, out_csv: Path, dst_crs: str | None = None) -> dict:
    dst_crs = dst_crs or CITY_PROJECTED_CRS[city]
    gdf = gpd.read_file(poi_path)
    if gdf.crs is None:
        raise ValueError(f"POI file has no CRS: {poi_path}")
    if "description" not in gdf.columns:
        raise KeyError(f"POI file must contain a description column: {poi_path}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf.to_crs(dst_crs)

    out = pd.DataFrame(
        {
            "poi_id": gdf.get("poi_id", pd.Series(range(len(gdf)), index=gdf.index)).astype(str).to_numpy(),
            "name": gdf.get("name", pd.Series("", index=gdf.index)).fillna("").astype(str).to_numpy(),
            "feature_easting": gdf.geometry.x.astype(float).to_numpy(),
            "feature_northing": gdf.geometry.y.astype(float).to_numpy(),
            "top_class": gdf.get("top_class", pd.Series("", index=gdf.index)).fillna("").astype(str).to_numpy(),
            "low_class": gdf.get("low_class", pd.Series("", index=gdf.index)).fillna("").astype(str).to_numpy(),
            "description": gdf["description"].fillna("").astype(str).to_numpy(),
        }
    )
    out = out[out["description"].str.len() > 0].reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    meta = {
        "city": city,
        "source_poi": str(poi_path),
        "out_csv": str(out_csv),
        "src_crs": gdf.crs.to_string(),
        "dst_crs": dst_crs,
        "rows": int(len(out)),
        "top_class_unique": int(out["top_class"].nunique()),
        "low_class_unique": int(out["low_class"].nunique()),
        "description_template": "benchmark_foursquare_description",
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark Foursquare POIs for CaLLiPer training.")
    parser.add_argument("--city", required=True, choices=sorted(CITY_PROJECTED_CRS))
    parser.add_argument("--poi", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--dst-crs", default=None)
    args = parser.parse_args()

    meta = prepare_city(args.city, args.poi, args.out_csv, args.dst_crs)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
