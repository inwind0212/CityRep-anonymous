#!/usr/bin/env python
"""Prepare benchmark-ready Mapillary manifests for adapted Urban2Vec.

The raw Mapillary metadata stores image ids and coordinates. This script links
those ids to local image files and writes a compact manifest that downstream
Urban2Vec preprocessing can consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "street_view" / "mapillary"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "street_view" / "mapillary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="City slug, e.g. singapore.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--metadata-name",
        default="metadata_selected.parquet",
        help="Metadata parquet inside the city raw Mapillary directory.",
    )
    parser.add_argument(
        "--out-name",
        default="manifest.parquet",
        help="Output manifest parquet name inside the processed city directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional deterministic head limit for quick smoke manifests.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional deterministic random fraction after image filtering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    city_dir = args.raw_root / args.city
    metadata_path = city_dir / args.metadata_name
    image_dir = city_dir / "images"
    out_dir = args.out_root / args.city
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata parquet: {metadata_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    df = pd.read_parquet(metadata_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    required = {"image_id", "lon", "lat"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["filepath"] = df["image_id"].map(lambda image_id: str(image_dir / f"{image_id}.jpg"))
    df["image_exists"] = df["filepath"].map(lambda path: Path(path).exists())
    before_filter = len(df)
    df = df[df["image_exists"]].copy()

    if args.sample_frac is not None:
        if not 0 < args.sample_frac <= 1:
            raise ValueError("--sample-frac must be in (0, 1].")
        df = df.sample(frac=args.sample_frac, random_state=args.seed).sort_index()

    keep_cols = [
        "image_id",
        "lon",
        "lat",
        "filepath",
        "captured_at",
        "compass_angle",
        "creator_id",
        "is_pano",
        "sequence_id",
        "foot",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    manifest = df[keep_cols].reset_index(drop=True)

    manifest_path = out_dir / args.out_name
    manifest_csv_path = out_dir / f"{manifest_path.stem}_preview.csv"
    meta_path = out_dir / "manifest_meta.json"
    manifest.to_parquet(manifest_path, index=False)
    manifest.head(1000).to_csv(manifest_csv_path, index=False)

    meta = {
        "city": args.city,
        "raw_metadata": str(metadata_path),
        "raw_image_dir": str(image_dir),
        "manifest": str(manifest_path),
        "n_metadata_rows_input": int(before_filter),
        "n_images_linked": int(len(manifest)),
        "n_missing_images": int(before_filter - len(manifest)),
        "limit": args.limit,
        "sample_frac": args.sample_frac,
        "seed": args.seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
