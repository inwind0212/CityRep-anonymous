#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Foursquare POIs with single primary categories for SPPE."
    )
    parser.add_argument("--input", required=True, help="Input Foursquare GeoParquet path.")
    parser.add_argument("--output", required=True, help="Output GPKG path.")
    parser.add_argument("--city", required=True)
    parser.add_argument(
        "--category-level",
        choices=["second", "leaf"],
        default="second",
        help="Use the primary Foursquare second-level category or primary leaf category as SPPE low_class.",
    )
    parser.add_argument("--keep-closed", action="store_true")
    parser.add_argument(
        "--keep-missing-category",
        action="store_true",
        help="Keep POIs without Foursquare category labels. Default drops them.",
    )
    parser.add_argument("--sample-n", type=int, default=0)
    return parser.parse_args()


def _to_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [str(value)]
    return [str(item).strip() for item in items if str(item).strip()]


def _primary_categories(category_labels: object, category_level: str) -> tuple[str, str, str]:
    labels = _to_list(category_labels)
    if not labels:
        return "Place", "Place", "Place"

    parts = [part.strip() for part in labels[0].split(">") if part.strip()]
    if not parts:
        return "Place", "Place", "Place"

    top = parts[0]
    second = parts[1] if len(parts) > 1 else parts[0]
    leaf = parts[-1]
    low = second if category_level == "second" else leaf
    return top, low, leaf


def _text(value: object, fallback: str) -> str:
    text = "" if value is None else str(value).strip()
    return text or fallback


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_parquet(input_path)
    total_count = int(len(gdf))
    if not args.keep_closed and "date_closed" in gdf.columns:
        gdf = gdf[gdf["date_closed"].isna()].copy()
    if not args.keep_missing_category and "fsq_category_labels" in gdf.columns:
        gdf = gdf[gdf["fsq_category_labels"].map(_to_list).map(bool)].copy()
    if args.sample_n and args.sample_n > 0 and len(gdf) > args.sample_n:
        gdf = gdf.sample(n=args.sample_n, random_state=42).sort_index().copy()

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom if geom is None or geom.geom_type == "Point" else geom.centroid
    )
    gdf = gdf[gdf.geometry.notna()].copy()

    names: list[str] = []
    poi_ids: list[str] = []
    top_classes: list[str] = []
    low_classes: list[str] = []
    leaf_classes: list[str] = []
    descriptions: list[str] = []

    for _, row in gdf.iterrows():
        name = _text(row.get("name"), "POI")
        top, low, leaf = _primary_categories(row.get("fsq_category_labels"), args.category_level)
        names.append(name)
        poi_ids.append(_text(row.get("fsq_place_id"), ""))
        top_classes.append(top)
        low_classes.append(low)
        leaf_classes.append(leaf)
        descriptions.append(f"This is a {low}, a type of {top}, named {name}.")

    out = gdf[["geometry"]].copy()
    out["poi_id"] = poi_ids
    out["name"] = names
    out["top_class"] = top_classes
    out["low_class"] = low_classes
    out["leaf_class"] = leaf_classes
    out["description"] = descriptions
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)
    out.to_file(output_path, layer="pois", driver="GPKG")

    meta = {
        "city": args.city,
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "total_input_rows": total_count,
        "output_rows": int(len(out)),
        "dropped_closed": not args.keep_closed,
        "dropped_missing_category": not args.keep_missing_category,
        "sample_n": int(args.sample_n),
        "category_level": args.category_level,
        "top_class_unique": int(out["top_class"].nunique(dropna=True)),
        "low_class_unique": int(out["low_class"].nunique(dropna=True)),
        "leaf_class_unique": int(out["leaf_class"].nunique(dropna=True)),
        "crs": out.crs.to_string() if out.crs else None,
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
