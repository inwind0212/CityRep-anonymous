#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(dirname "$HERE")}"
PY="${PY:-python}"
BOUNDARY_DIR="$ROOT/data/raw/boundaries"
ALPHAEARTH_ROOT="$ROOT/data/raw/alphaearth"
LOG_DIR="$ROOT/logs/raw_training"

mkdir -p "$LOG_DIR"

DEFAULT_CITIES=(london new_york singapore sydney mumbai nairobi jakarta cape_town)
if [[ $# -gt 0 ]]; then
  CITIES=("$@")
elif [[ -n "${BENCHMARK_CITIES:-}" ]]; then
  read -r -a CITIES <<< "$BENCHMARK_CITIES"
else
  CITIES=("${DEFAULT_CITIES[@]}")
fi

MAX_ALPHAEARTH_JOBS="${MAX_ALPHAEARTH_JOBS:-2}"
RUN_ALPHAEARTH="${RUN_ALPHAEARTH:-1}"
RUN_MAPILLARY="${RUN_MAPILLARY:-0}"
RUN_MANIFEST="${RUN_MANIFEST:-1}"

active_alphaearth_jobs() {
  jobs -pr | wc -l
}

wait_for_alphaearth_slot() {
  while [[ "$(active_alphaearth_jobs)" -ge "$MAX_ALPHAEARTH_JOBS" ]]; do
    sleep 30
  done
}

download_alphaearth_city() {
  local city="$1"
  local out="$ALPHAEARTH_ROOT/$city/alphaearth_2024.tif"
  local boundary="$BOUNDARY_DIR/$city.geojson"
  local log="$LOG_DIR/${city}_alphaearth.log"

  mkdir -p "$ALPHAEARTH_ROOT/$city"
  if [[ -s "$out" ]]; then
    echo "[$(date -Iseconds)] AlphaEarth exists: $city $out"
    return 0
  fi

  echo "[$(date -Iseconds)] AlphaEarth start: $city"
  "$PY" "$HERE/satellite/download_alphaearth_city.py" \
    --city "$city" \
    --boundary "$boundary" \
    --output "$out" \
    --overwrite \
    > "$log" 2>&1
  echo "[$(date -Iseconds)] AlphaEarth done: $city"
}

if [[ "$RUN_ALPHAEARTH" == "1" ]]; then
  for city in "${CITIES[@]}"; do
    wait_for_alphaearth_slot
    download_alphaearth_city "$city" &
  done
  wait
fi

if [[ "$RUN_MAPILLARY" == "1" ]]; then
  if [[ -z "${MAPILLARY_TOKEN:-}" ]]; then
    echo "MAPILLARY_TOKEN is required when RUN_MAPILLARY=1" >&2
    exit 1
  fi
  echo "[$(date -Iseconds)] Mapillary queue start: ${CITIES[*]}"
  "$HERE/street_view/run_mapillary_queue.sh" "${CITIES[@]}" \
    > "$LOG_DIR/mapillary_queue.log" 2>&1
  echo "[$(date -Iseconds)] Mapillary queue done"
fi

if [[ "$RUN_MANIFEST" == "1" ]]; then
  "$PY" "$HERE/check_raw_inputs.py" \
    > "$LOG_DIR/training_raw_manifest.log" 2>&1
  echo "[$(date -Iseconds)] Manifest updated: $ROOT/data/raw/training_raw_manifest.json"
fi
