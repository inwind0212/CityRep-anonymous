#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
from shapely.strtree import STRtree


LEGACY_ROOT = Path(__file__).resolve().parents[2]  # repository root
BOUNDARY_DIR = LEGACY_ROOT / "data" / "raw" / "boundaries"
WORLDPOP_DIR = LEGACY_ROOT / "data" / "worldpop"
OUT_ROOT = LEGACY_ROOT / "data" / "road_density"
OUTPUT_STEM = "road_density_2026"
OSM_EXTRACT_DATE = "2026-04-28"
TARGET_GRID_YEAR = "2024"

CITIES = ["london", "new_york", "singapore", "sydney", "mumbai", "nairobi", "jakarta", "cape_town"]

LOCAL_CRS = {
    "london": "EPSG:27700",
    "new_york": "EPSG:32618",
    "singapore": "EPSG:3414",
    "sydney": "EPSG:32756",
    "mumbai": "EPSG:32643",
    "nairobi": "EPSG:32737",
    "jakarta": "EPSG:32748",
    "cape_town": "EPSG:32734",
}

DRIVABLE_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}

NODATA = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WorldPop-grid road-density labels from OpenStreetMap roads.")
    parser.add_argument("--cities", nargs="*", default=CITIES, choices=CITIES)
    parser.add_argument("--overwrite-osm", action="store_true", help="Re-download OSM roads even if cached.")
    parser.add_argument("--overwrite-output", action="store_true", help="Overwrite existing road-density rasters.")
    parser.add_argument("--chunk-rows", type=int, default=48)
    parser.add_argument("--query-simplify-deg", type=float, default=0.001)
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between uncached Overpass requests.")
    return parser.parse_args()


def read_boundary(city: str) -> gpd.GeoDataFrame:
    path = BOUNDARY_DIR / f"{city}.geojson"
    boundary = gpd.read_file(path)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    return boundary.to_crs("EPSG:4326")


def highway_is_drivable(value: Any) -> bool:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return any(str(v) in DRIVABLE_HIGHWAYS for v in value)
    return str(value) in DRIVABLE_HIGHWAYS


def clean_osm_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    keep = [
        "osmid",
        "element_type",
        "highway",
        "name",
        "oneway",
        "lanes",
        "maxspeed",
        "surface",
        "bridge",
        "tunnel",
        "geometry",
    ]
    cols = [c for c in keep if c in gdf.columns]
    out = gdf[cols].copy()
    for col in out.columns:
        if col == "geometry":
            continue
        out[col] = out[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, tuple, dict, set, np.ndarray)) else v)
    return out


def fetch_or_load_roads(city: str, boundary: gpd.GeoDataFrame, out_dir: Path, overwrite: bool) -> gpd.GeoDataFrame:
    raw_path = out_dir / "osm_highways_raw.gpkg"
    roads_path = out_dir / "osm_drivable_roads.gpkg"
    if roads_path.is_file() and not overwrite:
        return gpd.read_file(roads_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    if raw_path.is_file() and not overwrite:
        raw = gpd.read_file(raw_path)
    else:
        geom = boundary.geometry.union_all()
        query_geom = geom.simplify(float(args.query_simplify_deg), preserve_topology=True)
        print(f"[osm] downloading {city} highways from Overpass")
        raw = ox.features_from_polygon(query_geom, tags={"highway": True})
        raw = raw.reset_index()
        clean_osm_columns(raw).to_file(raw_path, driver="GPKG")
        time.sleep(float(args.sleep))

    if raw.empty:
        raise ValueError(f"No OSM highway features found for {city}")

    raw = raw.set_crs(raw.crs or "EPSG:4326").to_crs("EPSG:4326")
    roads = raw[raw.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads[roads["highway"].map(highway_is_drivable)].copy()
    if "osmid" in roads.columns:
        roads = roads.drop_duplicates(subset=["osmid", "highway"])
    roads = roads.explode(index_parts=False, ignore_index=True)
    roads = gpd.clip(roads, boundary[["geometry"]])
    roads = roads[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    clean_osm_columns(roads).to_file(roads_path, driver="GPKG")
    return roads


def cell_polygons_for_rows(transform: rasterio.Affine, rows: np.ndarray, cols: np.ndarray) -> list:
    polygons = []
    for row, col in zip(rows, cols):
        left, top = transform * (int(col), int(row))
        right, bottom = transform * (int(col) + 1, int(row) + 1)
        polygons.append(box(min(left, right), min(bottom, top), max(left, right), max(bottom, top)))
    return polygons


def compute_density(city: str, roads_wgs84: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, out_dir: Path, chunk_rows: int) -> dict[str, Any]:
    target_path = WORLDPOP_DIR / city / "worldpop_2024.tif"
    if not target_path.exists():
        # The released road-density labels are on the same WorldPop 2024 grid.
        target_path = LEGACY_ROOT / "data" / "tasks" / f"{city}.road_density.2026" / "labels.tif"
    out_tif = out_dir / f"{OUTPUT_STEM}.tif"
    local_crs = LOCAL_CRS[city]

    roads_metric = roads_wgs84.to_crs(local_crs)
    roads_metric = roads_metric[roads_metric.length > 0].copy()
    road_geoms = np.array(list(roads_metric.geometry.values), dtype=object)
    tree = STRtree(road_geoms)

    with rasterio.open(target_path) as src:
        boundary_src = boundary.to_crs(src.crs)
        inside = ~geometry_mask(
            [boundary_src.geometry.union_all()],
            transform=src.transform,
            invert=False,
            out_shape=(src.height, src.width),
        )
        density = np.full((src.height, src.width), NODATA, dtype=np.float32)
        density[inside] = 0.0

        total_pairs = 0
        for row_start in range(0, src.height, chunk_rows):
            row_stop = min(src.height, row_start + chunk_rows)
            local_mask = inside[row_start:row_stop]
            rr, cc = np.where(local_mask)
            if len(rr) == 0:
                continue
            rr_abs = rr + row_start
            cells_wgs84 = cell_polygons_for_rows(src.transform, rr_abs, cc)
            cells_metric = gpd.GeoSeries(cells_wgs84, crs=src.crs).to_crs(local_crs)
            cell_geoms = np.array(list(cells_metric.values), dtype=object)
            cell_area_km2 = np.array([geom.area / 1_000_000.0 for geom in cell_geoms], dtype=np.float64)
            lengths = np.zeros(len(cell_geoms), dtype=np.float64)

            pairs = tree.query(cell_geoms, predicate="intersects")
            if pairs.size:
                total_pairs += int(pairs.shape[1])
                for cell_idx, road_idx in zip(pairs[0], pairs[1]):
                    inter = cell_geoms[int(cell_idx)].intersection(road_geoms[int(road_idx)])
                    if not inter.is_empty:
                        lengths[int(cell_idx)] += inter.length

            values = lengths / np.clip(cell_area_km2, 1e-12, None)
            density[rr_abs, cc] = values.astype(np.float32)
            print(f"[density] {city} rows {row_start}:{row_stop} cells={len(rr_abs)} pairs={total_pairs}")

        profile = src.profile.copy()
        profile.update(count=1, dtype="float32", nodata=NODATA, compress="lzw", BIGTIFF="YES")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(density, 1)

    valid = density != NODATA
    nonzero = valid & (density > 0)
    meta = {
        "city": city,
        "target": "road_density",
        "version": "2026",
        "source": "OpenStreetMap highway ways via OSMnx/Overpass",
        "osm_extract_date": OSM_EXTRACT_DATE,
        "target_grid_year": TARGET_GRID_YEAR,
        "highway_filter": sorted(DRIVABLE_HIGHWAYS),
        "definition": "sum of drivable road segment length inside each WorldPop grid cell divided by cell area",
        "unit": "meters_per_square_kilometer",
        "target_grid": str(target_path),
        "boundary": str(BOUNDARY_DIR / f"{city}.geojson"),
        "local_crs_for_length": local_crs,
        "output_tif": str(out_tif),
        "nodata": NODATA,
        "n_roads": int(len(roads_wgs84)),
        "valid_cells": int(valid.sum()),
        "nonzero_cells": int(nonzero.sum()),
        "zero_cell_rate": float(1.0 - nonzero.sum() / max(valid.sum(), 1)),
        "mean_density": float(density[valid].mean()),
        "median_density": float(np.median(density[valid])),
        "p95_density": float(np.percentile(density[valid], 95)),
        "max_density": float(density[valid].max()),
    }
    (out_dir / f"{OUTPUT_STEM}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    global args
    args = parse_args()
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OUT_ROOT / "_osmnx_cache")
    ox.settings.timeout = 900

    summaries = []
    for city in args.cities:
        out_dir = OUT_ROOT / city
        out_tif = out_dir / f"{OUTPUT_STEM}.tif"
        if out_tif.is_file() and not args.overwrite_output:
            print(f"[skip] {city}: {out_tif}")
            meta_path = out_dir / f"{OUTPUT_STEM}.meta.json"
            if meta_path.is_file():
                summaries.append(json.loads(meta_path.read_text()))
            continue
        print(f"[city] {city}")
        boundary = read_boundary(city)
        roads = fetch_or_load_roads(city, boundary, out_dir, overwrite=args.overwrite_osm)
        meta = compute_density(city, roads, boundary, out_dir, chunk_rows=args.chunk_rows)
        summaries.append(meta)
        print(json.dumps(meta, indent=2))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"{OUTPUT_STEM}_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
