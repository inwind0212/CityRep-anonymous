#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = ROOT / "data" / "raw" / "boundaries"
SINGAPORE_LAND_BOUNDARY = OUT_DIR / "singapore_planning_area_no_sea_2025.geojson"
KEEP_LARGEST_POLYGON_CITIES = {"singapore", "jakarta", "cape_town"}

CITY_QUERIES = {
    "london": "Greater London, England, United Kingdom",
    "new_york": "New York City, New York, United States",
    "singapore": "__local_singapore_land__",
    "sydney": "Sydney, New South Wales, Australia",
    "mumbai": [
        "Mumbai City District, Maharashtra, India",
        "Mumbai Suburban District, Maharashtra, India",
    ],
    "nairobi": "Nairobi, Kenya",
    "jakarta": [
        "Jakarta Barat, Daerah Khusus Ibukota Jakarta, Indonesia",
        "Jakarta Pusat, Daerah Khusus Ibukota Jakarta, Indonesia",
        "Jakarta Selatan, Daerah Khusus Ibukota Jakarta, Indonesia",
        "Jakarta Timur, Daerah Khusus Ibukota Jakarta, Indonesia",
        "Jakarta Utara, Daerah Khusus Ibukota Jakarta, Indonesia",
    ],
    "cape_town": "City of Cape Town, Western Cape, South Africa",
}

LAGOS_LGA_RELATIONS = {
    "Agege": 3718183,
    "Ajeromi/Ifelodun": 3718184,
    "Alimosho": 3718185,
    "Amuwo Odofin": 3718186,
    "Apapa": 3718187,
    "Eti Osa": 3718190,
    "Ikeja": 3718193,
    "Ikorodu": 3718194,
    "Kosofe": 3718195,
    "Lagos Island": 5404351,
    "Lagos Mainland": 5403871,
    "Mushin": 3718196,
    "Ojo": 3718197,
    "Oshodi/Isolo": 3718198,
    "Shomolu": 3718199,
    "Surulere": 3718200,
}


def fetch_boundary(query: str) -> dict:
    time.sleep(1.1)
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "polygon_geojson": 1,
    }
    headers = {
        "User-Agent": "urban-benchmark/0.1 (research data preparation)"
    }
    resp = requests.get(url, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    items = resp.json()
    if not items:
        raise RuntimeError(f"No boundary found for query: {query}")
    return items[0]


def fetch_osm_relation_polygon(name: str, relation_id: int) -> tuple[object, dict]:
    url = f"https://polygons.openstreetmap.fr/get_geojson.py?id={relation_id}&params=0"
    resp = requests.get(url, headers={"User-Agent": "urban-benchmark/0.1"}, timeout=120)
    resp.raise_for_status()
    geom = shape(resp.json())
    item = {
        "display_name": f"{name}, Lagos, Nigeria",
        "osm_id": relation_id,
        "osm_type": "relation",
        "place_id": None,
        "place_rank": 12,
        "class": "boundary",
        "type": "administrative",
    }
    return geom, item


def fetch_city_geometry(query: str | list[str]) -> tuple[object, list[dict]]:
    if query == "__local_singapore_land__":
        gdf = gpd.read_file(SINGAPORE_LAND_BOUNDARY).to_crs("EPSG:4326")
        geom = gdf.geometry.union_all()
        return geom, [
            {
                "display_name": "Singapore planning areas land boundary, no sea, 2025",
                "osm_id": None,
                "osm_type": "local",
                "place_id": None,
                "place_rank": None,
                "class": "boundary",
                "type": "land",
            }
        ]
    if query == "__lagos_lga_union__":
        geoms = []
        items = []
        for name, relation_id in LAGOS_LGA_RELATIONS.items():
            geom, item = fetch_osm_relation_polygon(name, relation_id)
            geoms.append(geom)
            items.append(item)
        return unary_union(geoms), items
    queries = query if isinstance(query, list) else [query]
    items = [fetch_boundary(part) for part in queries]
    geoms = [shape(item["geojson"]) for item in items]
    geom = geoms[0] if len(geoms) == 1 else unary_union(geoms)
    return geom, items


def maybe_keep_largest_polygon(city: str, geom):
    if city not in KEEP_LARGEST_POLYGON_CITIES:
        return geom, False
    if geom.geom_type != "MultiPolygon":
        return geom, False
    largest = max(geom.geoms, key=lambda g: g.area)
    return largest, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download city boundaries from OpenStreetMap/Nominatim.")
    parser.add_argument("--cities", nargs="+", default=list(CITY_QUERIES), choices=list(CITY_QUERIES))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing boundary files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for city in args.cities:
        query = CITY_QUERIES[city]
        out_path = OUT_DIR / f"{city}.geojson"
        meta_path = OUT_DIR / f"{city}.meta.json"
        if out_path.exists() and meta_path.exists() and not args.overwrite:
            print(f"[skip] {city}: {out_path}")
            continue
        geom, items = fetch_city_geometry(query)
        geom, kept_largest = maybe_keep_largest_polygon(city, geom)
        display_name = " | ".join(str(item.get("display_name")) for item in items)
        gdf = gpd.GeoDataFrame(
            [
                {
                    "city": city,
                    "query": json.dumps(query) if isinstance(query, list) else query,
                    "display_name": display_name,
                    "osm_id": ",".join(str(item.get("osm_id")) for item in items),
                    "osm_type": ",".join(str(item.get("osm_type")) for item in items),
                    "place_rank": ",".join(str(item.get("place_rank")) for item in items),
                    "type": ",".join(str(item.get("type")) for item in items),
                    "class": ",".join(str(item.get("class")) for item in items),
                    "kept_largest_polygon_only": kept_largest,
                    "geometry": geom,
                }
            ],
            crs="EPSG:4326",
        )
        gdf.to_file(out_path, driver="GeoJSON")

        meta = {
            "city": city,
            "query": query,
            "source": "OpenStreetMap Nominatim",
            "response": {
                "items": [
                    {
                        "display_name": item.get("display_name"),
                        "osm_id": item.get("osm_id"),
                        "osm_type": item.get("osm_type"),
                        "place_id": item.get("place_id"),
                        "place_rank": item.get("place_rank"),
                        "class": item.get("class"),
                        "type": item.get("type"),
                    }
                    for item in items
                ],
                "kept_largest_polygon_only": kept_largest,
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[saved] {city}: {out_path}")


if __name__ == "__main__":
    main()
