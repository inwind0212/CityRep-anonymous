#!/usr/bin/env bash
# Download the released embeddings from Kaggle and link them to the paths
# listed in baselines/registry/embedding_manifest.csv.
#
# The full dataset is about 52 GB. Downstream task data are already included
# in data/tasks/, so this is only needed to re-run the baseline evaluations.
set -euo pipefail

cd "$(dirname "$0")"

DATASET="${CITYREP_DATASET:-cityrep/cityrep}"
DEST="${CITYREP_DOWNLOAD_DIR:-downloads/cityrep}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "The Kaggle CLI is required: pip install kaggle (then configure your API token)." >&2
  exit 1
fi

mkdir -p "$DEST" baselines/artifacts
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip

"${CITYREP_PYTHON:-python}" - "$DEST" <<'PYTHON'
from pathlib import Path
import sys
import zipfile

source = Path(sys.argv[1]).resolve()
artifacts = Path("baselines/artifacts")

# Kaggle extracts the outer dataset archive; each model is a separate ZIP.
for model_zip in sorted((source / "embeddings").glob("*.zip")):
    output = model_zip.with_suffix("")
    print(f"extracting {model_zip.name}", flush=True)
    with zipfile.ZipFile(model_zip) as archive:
        for entry in archive.infolist():
            target = output / entry.filename
            if entry.is_dir() or (target.is_file() and target.stat().st_size == entry.file_size):
                continue
            archive.extract(entry, output)

roots = [source / "baselines/artifacts"]
roots.extend(sorted(source.glob("embeddings/*/baselines/artifacts")))
models = {model.name: model for root in roots if root.is_dir()
          for model in root.iterdir() if model.is_dir()}
if not models:
    raise SystemExit(f"No model artifacts found under {source}; check the Kaggle dataset layout.")

for name, model in sorted(models.items()):
    target = artifacts / name
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        print(f"keeping existing {target}")
        continue
    target.symlink_to(model, target_is_directory=True)
    print(f"linked {target}")
PYTHON
