# Raw input preparation

Scripts that download and preprocess the raw inputs used to build the baseline
embeddings. The downstream task data are already included in `data/tasks/` and
do not need to be rebuilt.

All scripts are run from the repository root and write under `data/raw/` and
`data/processed/` (both are git-ignored). Besides the main environment, they
need:

```bash
pip install duckdb mapillary osmnx requests pyyaml
```

## Inputs per baseline

| Input | Used by | Scripts |
|---|---|---|
| City boundaries | all inputs | `boundaries/download_city_boundaries.py` |
| Foursquare Open Source Places, release 2025-02-06 | AETHER, CaLLiPer, Space2Vec, Place2Vec, Urban2Vec, MuseCL, CityFM | `poi/download_foursquare_city_poi.py`, `poi/prepare_*.py` |
| Mapillary street-view images | Urban2Vec, MuseCL | `street_view/*.py`, `street_view/run_mapillary_queue.sh` |
| OpenStreetMap drivable roads, extract 2026-04-28 | CityFM | `osm/build_road_density_from_osm.py` |
| AlphaEarth annual embeddings, 2024 | AlphaEarth, AETHER | `satellite/download_alphaearth_city.py` |
| TESSERA embeddings, 2024 | TESSERA | `satellite/download_tessera_city.py` |
| Google Maps Static satellite tiles, zoom 17 | MuseCL | `satellite/download_google_static_tiles.py` |

`metadata/inputs/<city>/` records what was used for each city: the boundary,
the Foursquare snapshot and POI count, the OSM extract settings, the
AlphaEarth source, and the exact Mapillary images (ids and capture times) and
Google Static tiles used for Urban2Vec and MuseCL, with the sampling
parameters. Each script documents its options with `--help`.

## Steps

1. Boundaries. Copy the boundaries used in the paper,

   ```bash
   mkdir -p data/raw/boundaries
   for c in london new_york singapore sydney mumbai nairobi jakarta cape_town; do
     cp metadata/inputs/$c/boundary.geojson data/raw/boundaries/$c.geojson
   done
   ```

   or query them again from OpenStreetMap Nominatim with
   `boundaries/download_city_boundaries.py`. Singapore uses the land area of
   the 2025 planning-area layer, which has to be placed at
   `data/raw/boundaries/singapore_planning_area_no_sea_2025.geojson` first.

2. POIs. Clip Foursquare Open Source Places to each city and derive the
   primary second-level category and a short description per POI:

   ```bash
   python data_preparation/poi/download_foursquare_city_poi.py london --release 2025-02-06
   python data_preparation/poi/prepare_sppe_poi.py --city london \
     --input data/raw/poi/foursquare/london/fsq_os_places_2025-02-06.parquet \
     --output data/processed/poi/sppe/london/london_foursquare_primary_second.gpkg
   ```

   `prepare_calliper_poi.py`, `prepare_space2vec_poi.py`, and
   `prepare_aether_poi.py` convert the POIs into the input format of the
   corresponding models.

3. Street view. Requires a Mapillary access token in `MAPILLARY_TOKEN`. To
   download only the images used in the paper:

   ```bash
   python data_preparation/street_view/download_mapillary_city.py --city london \
     --metadata-parquet metadata/inputs/london/mapillary_images_urban2vec.csv.gz \
     --out-root data/raw/street_view/mapillary --download-images --image-res 1024
   ```

   `run_mapillary_queue.sh` instead collects all flat images inside the
   boundary (100-150 GB per city). `prepare_mapillary_manifest.py` and
   `sample_manifest_by_h3.py` then keep at most 10 images per H3 r8 cell,
   preferring 2024 and 2023 captures (seed 42).

4. Roads. Downloads OSM drivable highways through OSMnx/Overpass and writes
   `data/road_density/<city>/osm_highways_raw.gpkg`:

   ```bash
   python data_preparation/osm/build_road_density_from_osm.py --cities london
   ```

   OSM changes over time; the extract date and highway classes used in the
   paper are in `metadata/inputs/<city>/osm_roads.json`.

5. Satellite inputs. AlphaEarth tiles are read from the public Source
   Cooperative index (`tge-labs/aef`). TESSERA uses the official GeoTessera
   package, which needs Python 3.12. The Google Static tiles need an API key
   in `GOOGLE_MAPS_API_KEY`; pass
   `--regions metadata/inputs/<city>/google_static_tiles_musecl.csv.gz` to
   download the tiles used in the paper.

`run_raw_downloads.sh` runs the AlphaEarth downloads for all cities and,
with `RUN_MAPILLARY=1`, the street-view queue. `check_raw_inputs.py` writes
`data/raw/training_raw_manifest.json` with the availability and size of every
raw input.
