#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$HERE/../.." && pwd)}"
PY="${PY:-python}"
OUT_ROOT="$ROOT/data/raw/street_view/mapillary"
SCRIPT="$HERE/download_mapillary_city.py"
BOUNDARY_DIR="$ROOT/data/raw/boundaries"
LOG_DIR="$ROOT/logs/mapillary"

mkdir -p "$LOG_DIR"

if [[ -z "${MAPILLARY_TOKEN:-}" ]]; then
  echo "MAPILLARY_TOKEN is not set" >&2
  exit 1
fi

wait_for_file() {
  local path="$1"
  until [[ -f "$path" ]]; do
    sleep 60
  done
}

city_job_running() {
  local city="$1"
  pgrep -af "download_mapillary_city.py --city ${city} " >/dev/null 2>&1
}

wait_for_city_job() {
  local city="$1"
  while city_job_running "$city"; do
    echo "[$(date -Iseconds)] Waiting for existing job for $city"
    sleep 60
  done
}

run_download() {
  local city="$1"
  local boundary="$BOUNDARY_DIR/${city}.geojson"
  local city_dir="$OUT_ROOT/$city"
  local meta_parquet="$city_dir/metadata.parquet"

  mkdir -p "$city_dir"

  if [[ -f "$meta_parquet" ]]; then
    echo "[$(date -Iseconds)] Reuse existing metadata for $city: $meta_parquet"
  elif city_job_running "$city"; then
    echo "[$(date -Iseconds)] Metadata job already running for $city, waiting for $meta_parquet"
  else
    echo "[$(date -Iseconds)] Start metadata collection for $city"
    "$PY" "$SCRIPT" \
      --city "$city" \
      --boundary "$boundary" \
      --out-root "$OUT_ROOT" \
      > "$LOG_DIR/${city}_metadata.log" 2>&1
  fi

  wait_for_file "$meta_parquet"

  if city_job_running "$city"; then
    echo "[$(date -Iseconds)] Download job already running for $city, waiting for it to finish"
    wait_for_city_job "$city"
  else
    echo "[$(date -Iseconds)] Start full image download for $city"
    "$PY" "$SCRIPT" \
      --city "$city" \
      --metadata-parquet "$meta_parquet" \
      --out-root "$OUT_ROOT" \
      --download-images \
      --image-res 1024 \
      --workers 16 \
      > "$LOG_DIR/${city}_download.log" 2>&1
  fi

  echo "[$(date -Iseconds)] Finished $city"
}

CITIES=("$@")
if [[ ${#CITIES[@]} -eq 0 ]]; then
  if [[ -n "${MAPILLARY_CITIES:-}" ]]; then
    read -r -a CITIES <<< "$MAPILLARY_CITIES"
  else
    CITIES=(london new_york singapore sydney mumbai nairobi jakarta cape_town)
  fi
fi

for city in "${CITIES[@]}"; do
  if ! run_download "$city"; then
    echo "[$(date -Iseconds)] Failed $city, continuing to next city"
  fi
done
