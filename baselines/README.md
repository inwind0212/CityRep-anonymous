# Baselines

`registry/embedding_manifest.csv` lists the released embeddings of the 11
baselines, one row per model, city, and task, together with the form in which
each embedding is aligned to the task units. `../download.sh` downloads the
embeddings from Kaggle and links them under `artifacts/<model>/`, where the
manifest expects them.

The scripts that download and prepare the raw inputs of the baselines are in
`../data_preparation/`, and the inputs used for each city are recorded in
`../metadata/inputs/`.

H3 alignment follows Sec. 3.2 of the paper. CityFM and Place2Vec first
aggregate entity embeddings to H3 resolution-8 cells; Urban2Vec and MuseCL
provide region embeddings on that support. For areal task units, the default
`region_method=area_mean` takes an overlap-area-weighted mean over the covered
H3 cells, estimated by sub-point sampling within the task unit. Point tasks
use the containing H3 cell. Units without a matching representation receive
zero vectors. The manifest's `alignment_policy` describes this default;
selecting `region_method=centroid` explicitly uses only the cell at the unit
centre for areal tasks.
