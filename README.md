# Eval2Rn-Public

Data processing and modelling notebooks of the EVAL2Rn study on machine learning-based prediction of indoor radon concentrations and high-radon detection in Switzerland.

The notebooks are published without their outputs, as these contain confidential radon measurements.

## Environment

The analyses use Python 3.12 and the package versions pinned in `uv.lock`. With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync
uv run jupyter lab
```

## Data

The notebooks read their inputs from `data/raw/` at the root of the repository and write intermediate and final datasets to `data/processed/` and `data/final/`, models to `models/`, and figures and tables to `reports/`. These folders are not part of the repository.

| Folder in `data/raw/` | Source |
|---|---|
| `01_OFSP_radon_bdd` | Swiss national radon database, Federal Office of Public Health (FOPH) |
| `02_OFSP_radon_map` | Swiss radon map, FOPH |
| `03_OFS_regbl` | Federal Register of Buildings and Dwellings (RegBL), Federal Statistical Office (FSO) |
| `04_SWISSTOPO_geology` | Geology, swisstopo |
| `05_METESUISSE_climate` | Climate normals, MeteoSwiss |
| `06_OFEV_earthquakes` | Earthquakes, Federal Office for the Environment (FOEN) |
| `07_OFEV_uranium` | Uranium concentration, FOEN |
| `08_OFEV_hydrogeo_main_delimit` | Hydrogeology, main delimitations, FOEN |
| `10_OFEV_hydrogeo_watermass` | Hydrogeology, water masses, FOEN |
| `11_OFEV_hydrogeo_karst` | Hydrogeology, karst, FOEN |
| `12_SWISSTOPO_unconsolidated_soils` | Unconsolidated deposits, swisstopo |
| `13_SWISSTOPO_relief` | Digital height model, swisstopo |
| `14_OFEV_ecorforest_regions` | Forest site regions, FOEN |
| `15_OFS_area_statistics` | Area statistics, FSO |
| `17_ARE_zone_to_be_built` | Building zones, Federal Office for Spatial Development (ARE) |
| `18_MINERGIE_buildings` | Minergie-certified buildings |

All sources are publicly available except the national radon database, which is owned by the FOPH and subject to access restrictions: access requires a written request to the FOPH. Country boundaries and lakes for the maps are downloaded automatically from Natural Earth.

## Notebooks

Run the notebooks in the order of their numbers:

1. `notebooks/01_data-processing/`: processing of the raw datasets and their merge into the consolidated dataset, building by building (EGID).
2. `notebooks/02_modelling/`: data preparation and building-level split (0.1), encoding (0.2), model training and evaluation (0.3–0.5.4), and case studies (0.6.x, 0.7). Shared evaluation code is in `evaluation.py`.

## License

Apache License 2.0, see `LICENSE`.
