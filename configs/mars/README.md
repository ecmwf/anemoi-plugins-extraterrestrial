# ARCO-Mars Dataset Configs for Anemoi

Recipe configs for creating anemoi-datasets from the
[ARCO-Mars](https://arxiv.org/abs/2606.21701) unified cloud-optimized
Mars atmosphere reanalysis archive (Bhattacharya 2026).

## Datasets

| Config | Source | Grid | Levels | Mars Years | Steps/sol | DOI |
|---|---|---|---|---|---|---|
| `macda.yaml` | [ARCO-MACDA](https://huggingface.co/datasets/ananyo01/ARCO-MACDA) | 36x72 (5.0 deg) | 35 sigma | MY 24-35 | 12 | [10.57967/hf/8771](https://doi.org/10.57967/hf/8771) |
| `openmars.yaml` | [ARCO-OpenMARS](https://huggingface.co/datasets/ananyo01/ARCO-OpenMars) | 36x72 (5.0 deg) | 35 sigma | MY 24-35 | 12 | [10.57967/hf/8741](https://doi.org/10.57967/hf/8741) |
| `emars.yaml` | [ARCO-EMARS](https://huggingface.co/datasets/ananyo01/ARCO-EMARS) | 36x60 (5x6 deg) | 28 hybrid | MY 24-33 | 24 | [10.57967/hf/8859](https://doi.org/10.57967/hf/8859) |

## Backends

Each dataset is available from three hosting backends, selectable per
recipe via the `backend:` argument on the `arcomars` source. The
canonical short name (`ARCO-MACDA`, `ARCO-OpenMars`, `ARCO-EMARS`)
is the same on all — the plugin translates it to the correct
backend-specific repository id.

| Backend | Value | Repo / path pattern | Client | Extra install |
|---|---|---|---|---|
| HuggingFace Datasets (default) | `hf` | `ananyo01/ARCO-*` | `fsspec[hf]` | `pip install .[mars]` |
| Earthmover / Arraylake | `earthmover` | `arco-planetary/ARCO-*` | `arraylake` | `pip install .[mars,mars-earthmover]` |
| Local copy on disk | `local` | `local_path` / `$ARCO_MARS_LOCAL_ROOT` | `xarray` | `pip install .[mars]` |

The Earthmover backend follows the client pattern from
[ARCO-Mars-Examples](https://github.com/GalacticBobster/ARCO-Mars-Examples)
(`arraylake.Client().get_repo(...).readonly_session("main").store`).
It requires Arraylake credentials in the environment.

### Local backend

Use `backend: local` when you have already downloaded the ARCO-Mars
Zarr store(s) to disk (e.g. from HuggingFace) and want to build a
dataset without any network access. Point at the store with the
`local_path:` argument, which accepts either:

- the `.zarr` store itself (`/data/mars/macda_combined.zarr`), or
- a directory containing it (`/data/mars`) — in which case the
  dataset's default store filename is appended automatically:

| Dataset | Default local filename |
|---|---|
| ARCO-MACDA | `macda_combined.zarr` |
| ARCO-OpenMars | `openmars_unified.zarr` |
| ARCO-EMARS | `emars_combined.zarr` |

If `local_path` is omitted, the `$ARCO_MARS_LOCAL_ROOT` environment
variable is used as the base directory instead. The store is opened
directly with `xarray.open_zarr`, so the on-disk layout only needs to
be the downloaded `.zarr` store(s) — no HuggingFace-style directory
nesting is required.

### Store selection per dataset

Store / group selection is fixed per canonical dataset so recipes stay
backend-agnostic:

| Dataset | HF store | Earthmover group |
|---|---|---|
| ARCO-MACDA | `macda_combined.zarr` | root (`""`) |
| ARCO-OpenMars | `openmars_unified.zarr` | `my24` + `my28` merged into one continuous dataset |
| ARCO-EMARS | `emars_combined.zarr` | `mean` (ensemble mean) |

### Example recipe snippets

```yaml
# Default HuggingFace backend
input:
  join:
    - arcomars:
        dataset: ARCO-MACDA

# Earthmover backend
input:
  join:
    - arcomars:
        dataset: ARCO-EMARS
        backend: earthmover

# Local copy backend (predownloaded zarr on disk)
input:
  join:
    - arcomars:
        dataset: ARCO-MACDA
        backend: local
        local_path: /data/mars/macda_combined.zarr
```

## Time coordinate

Mars reanalysis data uses Mars sols as the native time coordinate.
The plugin maps each time step onto a **synthetic regular Earth-datetime grid**
(`step i -> 2000-01-01 + i * frequency`) so the anemoi pipeline, which
requires a fixed frequency, works unchanged. The frequency matches the
dataset's natural cadence:

| Dataset | Steps/sol | Mars cadence | Synthetic frequency | End date |
|---|---|---|---|---|
| MACDA | 12 | 2 Mars hours | `2h` | 2022-01-04T22:00 |
| OpenMARS | 12 | 2 Mars hours | `2h` | 2020-01-15T22:00 |
| EMARS | 24 | 1 Mars hour | `1h` | 2014-08-04T23:00 |

## Variable naming

Raw dataset variables are renamed to standard meteorological short names
following ECMWF / CF conventions where an Earth analogue exists:

| Standard name | MACDA raw | OpenMARS raw | EMARS raw | Description | Units |
|---|---|---|---|---|---|
| `t` | `temp` | `MY*_temp` | `anal_mean_T` | Temperature | K |
| `u` | `uwind` | `MY*_u` | `anal_mean_U` | Zonal wind | m/s |
| `v` | `vwind` | `MY*_v` | `anal_mean_V` | Meridional wind | m/s |
| `sp` | `psurf` | `MY*_ps` | `anal_mean_ps` | Surface pressure | Pa |
| `skt` | `tsurf` | `MY*_tsurf` | - | Surface temperature | K |
| `z` | `geop` | - | - | Geopotential | m2/s2 |
| `w` | `omega` | - | - | Vertical velocity | Pa/s |
| `co2ice` | `co2ice` | `MY*_co2ice` | - | Surface CO2 ice | kg/m2 |
| `tcdo` | `coldust` | `MY*_dustcol` | - | Column dust opacity | 1 |
| `dust` | `dustmmr` | - | - | Dust mass mixing ratio | 1 |
| `ssrd` | `swflux` | - | - | Surface solar radiation down | W/m2 |
| `strd` | `lwflux` | - | - | Surface thermal radiation down | W/m2 |
| `z_sfc` | - | - | `anal_mean_Surface_geopotential` | Surface geopotential | m2/s2 |

OpenMARS era-prefixed variables (`MY24-27_*`, `MY28-35_*`) are automatically
merged into single continuous variables by filling NaN gaps across eras.

Vertical levels are replaced with 1-indexed integers (1 = near-surface,
N = top of atmosphere).

## Forcings

Computed forcing parameters follow the design of Roy et al. (2026,
[arXiv:2605.28851](https://arxiv.org/abs/2605.28851)) and the standard
anemoi Earth forcings:

| Parameter | Type | Description |
|---|---|---|
| `cos_latitude` | Static | cos(latitude) |
| `sin_latitude` | Static | sin(latitude) |
| `cos_longitude` | Static | cos(longitude) |
| `sin_longitude` | Static | sin(longitude) |
| `cos_sol_of_year` | Dynamic | cos(sol-of-year angle), period = 1 Mars year (~668.6 sols) |
| `sin_sol_of_year` | Dynamic | sin(sol-of-year angle) |
| `cos_local_time` | Dynamic | cos(local sol-of-day), longitude-adjusted |
| `sin_local_time` | Dynamic | sin(local sol-of-day) |
| `cos_solar_longitude` | Dynamic | cos(Ls), areocentric solar longitude |
| `sin_solar_longitude` | Dynamic | sin(Ls), areocentric solar longitude |
| `insolation` | Dynamic | cos(Mars solar zenith angle), clipped to [0, 1] |

For **topography**, include the static surface altitude field directly from
the dataset (EMARS `z_sfc`, MACDA `z` at level 1) rather than computing it
as a forcing. See Roy et al. (2026) Section 4.3.1 for the Mars-Adapted
GraphCast forcing design.

## Usage

```bash
# Create a dataset from MACDA
anemoi-datasets create configs/mars/macda.yaml \
    mars-macda-hf-5p0-0024-0035-2h-v1.zarr

# Create from OpenMARS
anemoi-datasets create configs/mars/openmars.yaml \
    mars-openmars-hf-5p0-0024-0035-2h-v1.zarr

# Create from EMARS
anemoi-datasets create configs/mars/emars.yaml \
    mars-emars-hf-6p0-0024-0033-2h-v1.zarr

# Inspect the result
anemoi-datasets inspect mars-macda-hf-5p0-0024-0035-2h-v1.zarr
```

## References

- Bhattacharya (2026). *ARCO-Mars: A Unified Cloud-Optimized Archive of Mars
  Atmosphere Reanalysis.* [arXiv:2606.21701](https://arxiv.org/abs/2606.21701)
- Roy et al. (2026). *Towards a Foundation Model for the Martian Atmosphere.*
  [arXiv:2605.28851](https://arxiv.org/abs/2605.28851)
- Montabone et al. (2014). *The Mars Analysis Correction Data Assimilation
  (MACDA) dataset v1.0.* Geoscience Data Journal, 1(2), 129-139.
- Holmes, Lewis & Patel (2020). *OpenMARS: a global record of martian weather
  from 1999 to 2015.* Planetary and Space Science, 188, 104962.
- Greybush et al. (2019). *The Ensemble Mars Atmosphere Reanalysis System
  (EMARS) version 1.0.* Geoscience Data Journal, 6(2), 137-150.
