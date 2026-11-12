#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Foursquare city POIs into an AETHER-compatible POI file."
    )
    parser.add_argument("--input", required=True, help="Input Foursquare GeoParquet path.")
    parser.add_argument("--output", required=True, help="Output GPKG path.")
    parser.add_argument(
        "--city",
        required=True,
        help="City id used in metadata and synthesized descriptions.",
    )
    parser.add_argument(
        "--keep-closed",
        action="store_true",
        help="Keep POIs with date_closed set. Default drops them.",
    )
    parser.add_argument(
        "--keep-missing-category",
        action="store_true",
        help="Keep POIs without Foursquare category labels. Default drops them.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help="Optional deterministic row cap for smoke runs.",
    )
    parser.add_argument(
        "--description-template",
        default="simple_sentence_v3",
        choices=["simple_sentence_v3", "aether_example_v1"],
        help=(
            "Description text template. simple_sentence_v3 keeps the existing benchmark "
            "template; aether_example_v1 follows the wording in AETHER's poi_example.geojson."
        ),
    )
    return parser.parse_args()


def _normalize_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _to_label_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        try:
            items = list(value)
        except TypeError:
            items = [str(value)]
    labels = []
    for item in items:
        text = str(item).strip()
        if text:
            labels.append(text)
    return labels


def _build_description(
    name: str,
    category_labels: object,
    locality: object,
    country: object,
    template: str,
) -> tuple[str, str, str]:
    labels = _to_label_list(category_labels)
    top_classes: list[str] = []
    leaf_classes: list[str] = []
    for label in labels:
        parts = [part.strip() for part in label.split(">") if part.strip()]
        if not parts:
            continue
        top_classes.append(parts[0])
        leaf_classes.append(parts[-1])

    top_unique = list(dict.fromkeys(top_classes))
    leaf_unique = list(dict.fromkeys(leaf_classes))

    top_class = ", ".join(top_unique) if top_unique else "Place"
    low_class = ", ".join(leaf_unique) if leaf_unique else "Place"
    if template == "aether_example_v1":
        description = f"A place of {low_class}, a type of {top_class}, named {name}."
    elif labels:
        description = f"This is a {low_class}, a type of {top_class}, named {name}."
    else:
        description = f"This is a Place, a type of Place, named {name}."
    return description, top_class, low_class


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
        gdf = gdf[gdf["fsq_category_labels"].map(_to_label_list).map(bool)].copy()

    if args.sample_n and args.sample_n > 0 and len(gdf) > args.sample_n:
        gdf = gdf.sample(n=args.sample_n, random_state=42).sort_index().copy()

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom if geom is None or geom.geom_type == "Point" else geom.centroid
    )
    gdf = gdf[gdf.geometry.notna()].copy()

    names = []
    descriptions = []
    top_classes = []
    low_classes = []
    poi_ids = []

    for _, row in gdf.iterrows():
        name = _normalize_text(row.get("name"), "POI")
        description, top_class, low_class = _build_description(
            name=name,
            category_labels=row.get("fsq_category_labels"),
            locality=row.get("locality"),
            country=row.get("country"),
            template=args.description_template,
        )
        names.append(name)
        descriptions.append(description)
        top_classes.append(top_class)
        low_classes.append(low_class)
        poi_ids.append(_normalize_text(row.get("fsq_place_id"), ""))

    prepared = gdf[["geometry"]].copy()
    prepared["poi_id"] = poi_ids
    prepared["name"] = names
    prepared["top_class"] = top_classes
    prepared["low_class"] = low_classes
    prepared["description"] = descriptions
    prepared = prepared.set_geometry("geometry")
    prepared = gpd.GeoDataFrame(prepared, geometry="geometry", crs=gdf.crs)

    prepared.to_file(output_path, layer="pois", driver="GPKG")

    meta = {
        "city": args.city,
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "total_input_rows": total_count,
        "output_rows": int(len(prepared)),
        "dropped_closed": False if args.keep_closed else True,
        "dropped_missing_category": False if args.keep_missing_category else True,
        "sample_n": int(args.sample_n),
        "crs": prepared.crs.to_string() if prepared.crs else None,
        "description_template_version": args.description_template,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
