#!/usr/bin/env python
"""Sample a Mapillary manifest with a cap per H3 cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = ROOT / "data" / "processed" / "street_view" / "mapillary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--h3-res", type=int, default=8)
    parser.add_argument("--max-per-h3", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "temporal_priority"],
        default="temporal_priority",
    )
    parser.add_argument(
        "--priority-years",
        default="2024,2023",
        help="Comma-separated years to prioritize before falling back to older images.",
    )
    return parser.parse_args()


def _parse_priority_years(raw: str) -> list[int]:
    years: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    return years


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or DEFAULT_PROCESSED_ROOT / args.city / "manifest.parquet"
    df = pd.read_parquet(manifest_path)
    df = df.copy()
    df["h3_id"] = [
        h3.latlng_to_cell(float(lat), float(lon), int(args.h3_res))
        for lat, lon in zip(df["lat"].to_numpy(), df["lon"].to_numpy())
    ]
    priority_years = _parse_priority_years(args.priority_years)

    if args.sampling_strategy == "temporal_priority":
        captured = pd.to_datetime(df["captured_at"], unit="ms", errors="coerce", utc=True)
        capture_year = captured.dt.year.fillna(-1).astype(int)
        priority_rank = np.full(len(df), len(priority_years), dtype=np.int64)
        for idx, year in enumerate(priority_years):
            priority_rank[capture_year.to_numpy() == int(year)] = idx
        df["capture_year"] = capture_year
        df["priority_rank"] = priority_rank
        grouped = df.groupby("h3_id", sort=True, group_keys=False)
        sampled = grouped.apply(
            lambda x: x.sort_values(
                ["priority_rank", "capture_year", "captured_at", "image_id"],
                ascending=[True, False, False, True],
            ).head(int(args.max_per_h3))
        ).reset_index(drop=True)
    else:
        grouped = df.groupby("h3_id", sort=True, group_keys=False)
        sampled = grouped.apply(
            lambda x: x.sample(n=min(len(x), int(args.max_per_h3)), random_state=args.seed)
        ).reset_index(drop=True)
    sampled = sampled.sort_values(["h3_id", "image_id"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.out, index=False)
    sampled.head(1000).to_csv(args.out.with_name(args.out.stem + "_preview.csv"), index=False)

    per_h3 = sampled.groupby("h3_id").size()
    meta = {
        "city": args.city,
        "manifest": str(manifest_path),
        "out": str(args.out),
        "h3_res": int(args.h3_res),
        "max_per_h3": int(args.max_per_h3),
        "seed": int(args.seed),
        "sampling_strategy": args.sampling_strategy,
        "priority_years": priority_years,
        "n_input_images": int(len(df)),
        "n_output_images": int(len(sampled)),
        "n_h3_cells_output": int(sampled["h3_id"].nunique()),
        "per_h3_min": int(per_h3.min()) if len(per_h3) else 0,
        "per_h3_max": int(per_h3.max()) if len(per_h3) else 0,
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
