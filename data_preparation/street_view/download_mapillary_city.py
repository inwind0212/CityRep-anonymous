#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geopandas as gpd
import mapillary as mly
import pandas as pd
import requests
from tqdm import tqdm
from shapely.geometry import Point, shape


METADATA_COLUMNS = ["image_id", "lon", "lat", "captured_at", "compass_angle", "creator_id", "is_pano", "sequence_id", "foot"]
GRAPH_IMAGE_FIELDS = "id,geometry,captured_at,compass_angle,creator,is_pano,sequence"
GRAPH_IMAGE_LIMIT = 2000
GRAPH_SPLIT_MIN_SPAN = 1e-4
GRAPH_SPLIT_MAX_DEPTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Mapillary metadata and thumbnails for a city boundary.")
    parser.add_argument("--city", required=True)
    parser.add_argument("--boundary", type=Path, default=None)
    parser.add_argument("--bbox", type=str, default=None, help="west,south,east,north in EPSG:4326")
    parser.add_argument("--metadata-parquet", type=Path, default=None, help="Use an existing metadata table (parquet, or a csv such as metadata/inputs/<city>/mapillary_images_*.csv.gz) instead of querying Mapillary.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--image-type", choices=["flat", "pano", "all"], default="flat")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--image-res", type=int, choices=[256, 1024, 2048], default=1024)
    parser.add_argument("--max-images", type=int, default=0, help="0 means keep all metadata rows / download all selected rows.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--tile-deg",
        type=float,
        default=0.02,
        help="When using --boundary, query Mapillary in bbox tiles of this size (degrees).",
    )
    return parser.parse_args()


def load_boundary(boundary_path: Path) -> tuple[dict[str, Any], Any]:
    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.union_all()
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": json.loads(gpd.GeoSeries([geom], crs="EPSG:4326").to_json())["features"][0]["geometry"],
            }
        ],
    }
    return feature_collection, geom


def empty_metadata_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=METADATA_COLUMNS)


def metadata_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return empty_metadata_dataframe()
    df = pd.DataFrame(rows)
    for col in METADATA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[METADATA_COLUMNS].dropna(subset=["image_id", "lon", "lat"]).drop_duplicates(subset=["image_id"]).reset_index(drop=True)


def split_bbox(west: float, south: float, east: float, north: float) -> list[tuple[float, float, float, float]]:
    mid_x = (west + east) / 2
    mid_y = (south + north) / 2
    return [
        (west, south, mid_x, mid_y),
        (mid_x, south, east, mid_y),
        (west, mid_y, mid_x, north),
        (mid_x, mid_y, east, north),
    ]


def normalize_graph_feature(item: dict[str, Any]) -> dict[str, Any]:
    geom = item.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None]
    creator = item.get("creator") or {}
    return {
        "image_id": str(item.get("id")),
        "lon": coords[0],
        "lat": coords[1],
        "captured_at": item.get("captured_at"),
        "compass_angle": item.get("compass_angle"),
        "creator_id": creator.get("id"),
        "is_pano": item.get("is_pano"),
        "sequence_id": item.get("sequence"),
        "foot": item.get("foot"),
    }


def fetch_graph_bbox_rows(
    session: requests.Session,
    west: float,
    south: float,
    east: float,
    north: float,
    image_type: str,
    depth: int = 0,
) -> list[dict[str, Any]]:
    span = max(east - west, north - south)
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "fields": GRAPH_IMAGE_FIELDS,
        "image_type": image_type,
        "limit": GRAPH_IMAGE_LIMIT,
    }
    resp = session.get("https://graph.mapillary.com/images", params=params, timeout=120)
    if resp.status_code >= 400:
        if resp.status_code in {400, 500} and depth < GRAPH_SPLIT_MAX_DEPTH and span > GRAPH_SPLIT_MIN_SPAN:
            rows: list[dict[str, Any]] = []
            for child_bbox in split_bbox(west, south, east, north):
                rows.extend(fetch_graph_bbox_rows(session, *child_bbox, image_type=image_type, depth=depth + 1))
            return rows
        resp.raise_for_status()
    payload = resp.json()

    if "error" in payload:
        message = str(payload["error"].get("message", ""))
        if "reduce the amount of data" in message.lower() and depth < GRAPH_SPLIT_MAX_DEPTH and span > GRAPH_SPLIT_MIN_SPAN:
            rows: list[dict[str, Any]] = []
            for child_bbox in split_bbox(west, south, east, north):
                rows.extend(fetch_graph_bbox_rows(session, *child_bbox, image_type=image_type, depth=depth + 1))
            return rows
        raise RuntimeError(message or "Mapillary Graph API returned an unknown error")

    data = payload.get("data", [])
    if len(data) >= GRAPH_IMAGE_LIMIT and depth < GRAPH_SPLIT_MAX_DEPTH and span > GRAPH_SPLIT_MIN_SPAN:
        rows = []
        for child_bbox in split_bbox(west, south, east, north):
            rows.extend(fetch_graph_bbox_rows(session, *child_bbox, image_type=image_type, depth=depth + 1))
        return rows

    return [normalize_graph_feature(item) for item in data]


def generate_boundary_tiles(boundary_geom: Any, tile_deg: float) -> list[tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = boundary_geom.bounds
    tiles: list[tuple[float, float, float, float]] = []
    x_steps = max(1, math.ceil((maxx - minx) / tile_deg))
    y_steps = max(1, math.ceil((maxy - miny) / tile_deg))
    for ix in range(x_steps):
        west = minx + ix * tile_deg
        east = min(west + tile_deg, maxx)
        for iy in range(y_steps):
            south = miny + iy * tile_deg
            north = min(south + tile_deg, maxy)
            tile_poly = shape(
                {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [west, south],
                            [east, south],
                            [east, north],
                            [west, north],
                            [west, south],
                        ]
                    ],
                }
            )
            if boundary_geom.intersects(tile_poly):
                tiles.append((west, south, east, north))
    return tiles


def collect_tile_rows(
    west: float,
    south: float,
    east: float,
    north: float,
    image_type: str,
    boundary_geom: Any,
    seen_ids: set[str],
    session: requests.Session,
) -> list[dict[str, Any]]:
    try:
        raw_rows = fetch_graph_bbox_rows(session, west, south, east, north, image_type=image_type)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "warning": "skip_tile_query_error",
                    "bbox": [west, south, east, north],
                    "error": repr(exc),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        image_id = item["image_id"]
        lon, lat = item["lon"], item["lat"]
        if image_id in seen_ids or lon is None or lat is None:
            continue
        pt = Point(lon, lat)
        if not boundary_geom.covers(pt):
            continue
        seen_ids.add(image_id)
        rows.append(item)
    return rows


def write_metadata_checkpoint(
    rows: list[dict[str, Any]],
    out_path: Path,
    progress_path: Path,
    city: str,
    tiles_done: int,
    tiles_total: int,
) -> None:
    df = metadata_rows_to_dataframe(rows)
    df["filepath"] = pd.NA
    df.to_parquet(out_path, index=False)
    progress = {
        "city": city,
        "tiles_done": tiles_done,
        "tiles_total": tiles_total,
        "metadata_rows": int(len(df)),
        "checkpoint_parquet": str(out_path.resolve()),
    }
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_rows(df: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    if max_images <= 0 or len(df) <= max_images:
        return df.copy()
    return df.sample(n=max_images, random_state=seed).sort_values("image_id").reset_index(drop=True)


def fetch_thumb_url(image_id: str, resolution: int) -> str:
    return mly.interface.image_thumbnail(image_id, resolution)


def download_one(image_id: str, out_path: Path, resolution: int) -> tuple[str, str | None]:
    try:
        url = fetch_thumb_url(image_id, resolution)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return image_id, None
    except Exception as exc:  # noqa: BLE001
        return image_id, repr(exc)


def download_selected_images(
    selected: pd.DataFrame,
    images_dir: Path,
    resolution: int,
    workers: int,
    city: str,
) -> tuple[int, list[dict[str, str]]]:
    downloaded = 0
    download_errors: list[dict[str, str]] = []
    download_targets: list[tuple[str, Path]] = []
    for image_id in selected["image_id"]:
        out_path = images_dir / f"{image_id}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            downloaded += 1
            continue
        download_targets.append((image_id, out_path))

    max_in_flight = max(workers * 4, workers)
    target_iter = iter(download_targets)
    futures: dict[Any, tuple[str, Path]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while len(futures) < max_in_flight:
            try:
                image_id, out_path = next(target_iter)
            except StopIteration:
                break
            futures[ex.submit(download_one, image_id, out_path, resolution)] = (image_id, out_path)

        with tqdm(total=len(download_targets), desc=f"{city} mapillary") as pbar:
            while futures:
                fut = next(as_completed(futures))
                image_id, _ = futures.pop(fut)
                _, err = fut.result()
                if err is None:
                    downloaded += 1
                else:
                    download_errors.append({"image_id": image_id, "error": err})
                pbar.update(1)

                try:
                    next_image_id, next_out_path = next(target_iter)
                except StopIteration:
                    continue
                futures[ex.submit(download_one, next_image_id, next_out_path, resolution)] = (next_image_id, next_out_path)

    return downloaded, download_errors


def main() -> None:
    args = parse_args()
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise EnvironmentError("MAPILLARY_TOKEN is not set.")

    city_root = args.out_root / args.city
    images_dir = city_root / "images"
    city_root.mkdir(parents=True, exist_ok=True)
    if args.download_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    mly.interface.set_access_token(token)
    source_count = sum(x is not None for x in [args.boundary, args.bbox, args.metadata_parquet])
    if source_count != 1:
        raise ValueError("Provide exactly one of --boundary, --bbox, or --metadata-parquet.")

    meta_path = city_root / "metadata.parquet"
    meta_partial_path = city_root / "metadata.partial.parquet"
    progress_path = city_root / "metadata_progress.json"

    if args.metadata_parquet is not None:
        source = Path(args.metadata_parquet)
        metadata = pd.read_csv(source, dtype={"image_id": str, "sequence_id": str}) if source.suffix in {".csv", ".gz"} else pd.read_parquet(source)
        metadata = metadata_rows_to_dataframe(metadata.to_dict(orient="records"))
    elif args.bbox is not None:
        west, south, east, north = [float(x) for x in args.bbox.split(",")]
        with requests.Session() as session:
            session.params = {"access_token": token}
            rows = fetch_graph_bbox_rows(session, west, south, east, north, image_type=args.image_type)
        metadata = metadata_rows_to_dataframe(rows)
    else:
        _, boundary_geom = load_boundary(args.boundary)
        seen_ids: set[str] = set()
        rows: list[dict[str, Any]] = []
        tiles = generate_boundary_tiles(boundary_geom, tile_deg=args.tile_deg)
        tiles_total = len(tiles)
        checkpoint_every = 10
        with requests.Session() as session:
            session.params = {"access_token": token}
            for idx, (west, south, east, north) in enumerate(tiles, start=1):
                rows.extend(collect_tile_rows(west, south, east, north, args.image_type, boundary_geom, seen_ids, session))
                if idx % checkpoint_every == 0 or idx == tiles_total:
                    write_metadata_checkpoint(
                        rows=rows,
                        out_path=meta_partial_path,
                        progress_path=progress_path,
                        city=args.city,
                        tiles_done=idx,
                        tiles_total=tiles_total,
                    )
                    print(
                        json.dumps(
                            {
                                "city": args.city,
                                "tiles_done": idx,
                                "tiles_total": tiles_total,
                                "metadata_rows": len(rows),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        metadata = metadata_rows_to_dataframe(rows)
    metadata["filepath"] = pd.NA

    selected = choose_rows(metadata, max_images=args.max_images, seed=args.sample_seed)
    selected_ids = set(selected["image_id"].tolist())

    selected_path = city_root / "metadata_selected.parquet"
    metadata.to_parquet(meta_path, index=False)
    selected.to_parquet(selected_path, index=False)

    download_errors: list[dict[str, str]] = []
    downloaded = 0
    if args.download_images:
        downloaded, download_errors = download_selected_images(
            selected=selected,
            images_dir=images_dir,
            resolution=args.image_res,
            workers=args.workers,
            city=args.city,
        )

        selected = selected.copy()
        selected["filepath"] = selected["image_id"].map(lambda x: str((images_dir / f"{x}.jpg").resolve()) if (images_dir / f"{x}.jpg").exists() else pd.NA)
        selected.to_parquet(selected_path, index=False)

    meta = {
        "city": args.city,
        "boundary": str(args.boundary.resolve()) if args.boundary is not None else None,
        "bbox": args.bbox,
        "image_type": args.image_type,
        "n_metadata_images": int(len(metadata)),
        "n_selected_images": int(len(selected_ids)),
        "download_images": bool(args.download_images),
        "image_res": int(args.image_res),
        "n_downloaded_images": int(downloaded),
        "n_failed_downloads": int(len(download_errors)),
        "metadata_parquet": str(meta_path.resolve()),
        "selected_parquet": str(selected_path.resolve()),
    }
    if download_errors:
        err_path = city_root / "download_errors.json"
        err_path.write_text(json.dumps(download_errors, ensure_ascii=False, indent=2), encoding="utf-8")
        meta["download_errors_json"] = str(err_path.resolve())

    meta_path_json = city_root / "download_meta.json"
    meta_path_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
