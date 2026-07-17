# Regridded ARCO-Mars configs

Variants of the top-level `configs/mars/*.yaml` recipes that take an
**already-built** sibling anemoi dataset as their input (via the
built-in `anemoi_dataset` source) and pass it through a `regrid` step
onto an octahedral reduced Gaussian grid.

The pattern is:

```yaml
input:
  pipe:
    - anemoi_dataset:
        dataset: <name of the sibling dataset built from ../*.yaml>
    - regrid:
        matrix: <path to a pre-computed regridding matrix>
```

`../openmars_regridded.yaml` is the pre-existing canonical example
(left untouched); the configs in this directory follow the same
pattern but source from a prebuilt zarr instead of re-running the
`arcomars` source.

## Configs

| Config | Input dataset | Native grid | Target grid | Matrix |
|---|---|---|---|---|
| `macda.yaml` | `mars-macda-hf-5p0-0024-0035-2h-v3` | 36x72 (5.0 deg) | O24 | `5deg_to_o24.npz` |
| `combined.yaml` | `mars-openmars-macda-hf-5p0-0024-0035-2h-v3` | 36x72 (5.0 deg) | O24 | `5deg_to_o24.npz` |

Build order: build the sibling non-regrid dataset first (from
`../macda.yaml` / `../combined.yaml`), then build the regridded
variant from this directory.

An EMARS variant is not provided here because EMARS is on a
5.0 x 6.0 deg grid (36 x 60), which none of the shipped matrices
in `static/matrices/mars/` currently target.  Available matrices:
`5deg_to_o16.npz`, `5deg_to_o18.npz`, `5deg_to_o24.npz` (all 5-deg
inputs).

## Matrix path

The `regrid.matrix` field is repo-relative
(`static/matrices/mars/5deg_to_o24.npz`).  Resolve it against your
checkout root when running `anemoi-datasets create`, or replace with
an absolute path locally if your runner requires one.

## See also

* `../openmars_regridded.yaml` — the canonical example (do not modify).
* `../README.md` — dataset overview and backend documentation.
