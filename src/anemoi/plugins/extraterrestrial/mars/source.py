# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Anemoi-datasets source for ARCO Mars reanalysis stores.

Two hosting backends are supported (selectable per recipe via the
``backend:`` argument):

* ``hf`` (default) — HuggingFace Datasets (``ananyo01/ARCO-*``), opened
  via ``fsspec``.
* ``earthmover`` — Earthmover / Arraylake catalogue
  (``arco-planetary/ARCO-*``), opened via ``arraylake.Client``.  See
  https://github.com/GalacticBobster/ARCO-Mars-Examples for the
  upstream reference client usage.

Supports the following datasets hosted by ``ananyo01`` on HuggingFace:

* **ARCO-MACDA** (``macda_combined.zarr``)
  Variables (renamed to standard short names):
  ``t``, ``u``, ``v``, ``sp``, ``skt``, ``z``, ``w``,
  ``ssrd``, ``strd``, ``co2ice``, ``tcdo``, ``dust``

* **ARCO-OpenMars** (``openmars_unified.zarr``)
  Variables are era-prefixed with standard suffixes:
  ``MY24-27_t``, ``MY28-35_t``, ``MY24-27_u``, ``MY28-35_u``,
  ``MY24-27_v``, ``MY28-35_v``, ``MY24-27_sp``, ``MY28-35_sp``,
  ``MY24-27_skt``, ``MY28-35_skt``, ``MY24-27_co2ice``,
  ``MY28-35_co2ice``, ``MY24-27_tcdo``, ``MY28-35_tcdo``

Both stores use Zarr v3 format and encode time as fractional Mars
sols since Mars Year 24 Ls=0 (Northern Spring Equinox).  The native
sampling is every 1/12 sol (~2h 3m 18s Earth time), which is not a
regular Earth interval.  This plugin replaces the time axis with a
**synthetic regular 2h Earth grid** so that the anemoi-datasets
pipeline (which requires a fixed ``frequency``) works unchanged.
Time step *i* is mapped to ``epoch + i * 2h``.

Dates config
------------
The synthetic grid determines the ``start``, ``end``, and
``frequency`` for your recipe::

    dates:
      # MACDA (96 480 steps)
      start: '2000-01-01T00:00:00'
      end:   '2022-01-04T22:00:00'
      frequency: 2h

      # OpenMars unified (87 840 steps)
      start: '2000-01-01T00:00:00'
      end:   '2020-01-15T22:00:00'
      frequency: 2h

Example YAML recipe
-------------------
::

    input:
      join:
        # HuggingFace backend (default)
        - arcomars:
            dataset: ARCO-MACDA          # canonical short name
            param: [t, u, v, sp]

        # Earthmover / Arraylake backend
        - arcomars:
            dataset: ARCO-EMARS
            backend: earthmover
            param: [t, u, v]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from anemoi.datasets.create.sources.xarray import XarraySourceBase
from anemoi.datasets.create.sources.xarray_support import load_one
from anemoi.datasets.create.types import DateList

if TYPE_CHECKING:
    import xarray as xr

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic regular Earth-time grid
# ---------------------------------------------------------------------------
# The native Mars sampling is every 1/12 sol (~2h 3m 18s Earth time),
# which is not a regular Earth interval.  We assign each time step a
# position on a regular **2h Earth grid** so the pipeline can use
# ``frequency: 2h``.  Time step *i* maps to ``_GRID_EPOCH + i * 2h``.
#
# The epoch is an arbitrary round date chosen so that the grid values
# are easy to read and don't collide with real Earth reanalysis dates.
_GRID_EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")

# One Mars mean solar day ("sol") expressed in SI (Earth) days.
# 24 h 39 m 35.244 s  =  88 775.244 s
# (kept for reference / downstream utilities)
SOL_IN_EARTH_DAYS = 88_775.244 / 86_400.0  # ~1.02749125


def sols_to_regular_grid(n: int, frequency_h: int = 2) -> np.ndarray:
    """Create a regular Earth-datetime grid with *n* steps.

    Parameters
    ----------
    n : int
        Number of time steps.
    frequency_h : int
        Grid step size in hours.  ``2`` for MACDA / OpenMars (12
        steps/sol), ``1`` for EMARS (24 steps/sol).

    Returns
    -------
    np.ndarray
        Array of ``numpy.datetime64[ns]`` values evenly spaced,
        starting at :data:`_GRID_EPOCH`.
    """
    step = np.timedelta64(frequency_h, "h")
    return _GRID_EPOCH + np.arange(n, dtype="int64") * step


# ---------------------------------------------------------------------------
# Backend catalogue
# ---------------------------------------------------------------------------
# Each Mars reanalysis dataset is hosted on multiple backends:
#
# * **hf** — HuggingFace Datasets (``ananyo01/ARCO-*``), Zarr v3 stores
#   opened via ``fsspec``.
# * **earthmover** — Earthmover / Arraylake catalogue
#   (``arco-planetary/ARCO-*``), opened via ``arraylake.Client``.  Data
#   is organised into named *groups* within each repo.
#
# The catalogue below is keyed by a canonical short name (``ARCO-MACDA``,
# ``ARCO-OpenMars``, ``ARCO-EMARS``).  Recipes may pass either the short
# name or a fully-qualified backend-specific id (``ananyo01/ARCO-MACDA``
# or ``arco-planetary/ARCO-MACDA``); see :func:`_normalise_dataset`.

# Canonical short names for the three Mars reanalyses.
CANONICAL_MACDA = "ARCO-MACDA"
CANONICAL_OPENMARS = "ARCO-OpenMars"
CANONICAL_EMARS = "ARCO-EMARS"

_KNOWN_STORES: dict[str, dict] = {
    CANONICAL_MACDA: {
        "nlat": 36,
        "nlon": 72,
        "steps_per_sol": 12,
        "frequency_h": 2,  # synthetic grid step in hours
        "hf": {
            "repo": "ananyo01/ARCO-MACDA",
            "default": "macda_combined.zarr",
        },
        "earthmover": {
            "repo": "arco-planetary/ARCO-MACDA",
            "default_group": "",
        },
    },
    CANONICAL_OPENMARS: {
        "nlat": 36,
        "nlon": 72,
        "steps_per_sol": 12,
        "frequency_h": 2,
        "hf": {
            "repo": "ananyo01/ARCO-OpenMars",
            "default": "openmars_unified.zarr",
            "MY24-27": "openmars_MY24-27.zarr",
            "MY28-35": "openmars_MY28-35.zarr",
        },
        "earthmover": {
            "repo": "arco-planetary/ARCO-OpenMARS",
            # Earthmover stores the two eras as separate zarr groups.
            # We open both and merge them (identical logic to the HF
            # unified store) so the user sees one continuous dataset.
            "era_groups": ("my24", "my28"),
        },
    },
    CANONICAL_EMARS: {
        "nlat": 36,
        "nlon": 60,
        "steps_per_sol": 24,
        "frequency_h": 1,
        "hf": {
            "repo": "ananyo01/ARCO-EMARS",
            "default": "emars_combined.zarr",
        },
        "earthmover": {
            "repo": "arco-planetary/ARCO-EMARS",
            # EMARS on earthmover has 'mean' (ensemble mean) and 'sprd'
            # (ensemble spread) groups.  We default to the mean.
            "default_group": "mean",
        },
    },
}

# Reverse lookup: any known id (short or fully-qualified per backend)
# maps to the canonical short name.
_ID_TO_CANONICAL: dict[str, str] = {}
for _canon, _info in _KNOWN_STORES.items():
    _ID_TO_CANONICAL[_canon] = _canon
    _ID_TO_CANONICAL[_canon.lower()] = _canon
    for _backend in ("hf", "earthmover"):
        _repo = _info[_backend]["repo"]
        _ID_TO_CANONICAL[_repo] = _canon
        _ID_TO_CANONICAL[_repo.lower()] = _canon
# Common capitalisation variants for OpenMars vs OpenMARS.
_ID_TO_CANONICAL["ARCO-OpenMARS"] = CANONICAL_OPENMARS
_ID_TO_CANONICAL["ARCO-openmars"] = CANONICAL_OPENMARS


def _normalise_dataset(dataset: str) -> str:
    """Return the canonical short name for a dataset id.

    Accepts either the short name (``ARCO-MACDA``) or a fully-qualified
    id for any backend (``ananyo01/ARCO-MACDA``,
    ``arco-planetary/ARCO-MACDA``).  Matching is case-insensitive.

    Raises
    ------
    ValueError
        If the id is not recognised.
    """
    canon = _ID_TO_CANONICAL.get(dataset) or _ID_TO_CANONICAL.get(dataset.lower())
    if canon is None:
        raise ValueError(
            f"Unknown Mars dataset id '{dataset}'.  Known: "
            f"{sorted(set(_KNOWN_STORES))} (or fully-qualified backend ids)."
        )
    return canon


# Non-data variables that should be dropped before handing the
# dataset to the anemoi xarray field-list machinery.  These are
# auxiliary coordinate-like arrays that would otherwise confuse
# the ``CoordinateGuesser``.
_DROP_VARS = {
    "Ls",
    "MY_Ls",
    "MY24-27_Ls",
    "MY28-35_Ls",
    "MY24-27_MY",
    "MY28-35_MY",
}

# ---------------------------------------------------------------------------
# Variable rename maps  –  original name → standard short name
# ---------------------------------------------------------------------------
# Names follow ECMWF / CF conventions where an Earth analogue exists.
# Mars-specific quantities keep descriptive short names.
#
#   t      – temperature                    (CF: air_temperature)
#   u      – zonal (eastward) wind          (CF: eastward_wind)
#   v      – meridional (northward) wind    (CF: northward_wind)
#   sp     – surface pressure               (ECMWF: sp)
#   skt    – skin (surface) temperature     (ECMWF: skt)
#   z      – geopotential                   (ECMWF: z)
#   w      – pressure vertical velocity     (CF: lagrangian_tendency_of_air_pressure)
#   ssrd   – surface solar radiation down   (ECMWF: ssrd)
#   strd   – surface thermal radiation down (ECMWF: strd)
#   co2ice – surface CO₂ ice               (Mars-specific)
#   tcdo   – total column dust opacity      (Mars-specific, cf. ECMWF "tc" prefix)
#   dust   – dust mass mixing ratio         (Mars-specific)

_RENAME_MACDA: dict[str, str] = {
    "temp": "t",
    "uwind": "u",
    "vwind": "v",
    "psurf": "sp",
    "tsurf": "skt",
    "geop": "z",
    "omega": "w",
    "swflux": "ssrd",
    "lwflux": "strd",
    "co2ice": "co2ice",
    "coldust": "tcdo",
    "dustmmr": "dust",
}

# OpenMars unified: maps the raw variable suffix (after stripping
# the era prefix) to a standard short name.  The two eras are
# merged into a single continuous variable per quantity.
_RENAME_OPENMARS_SUFFIX: dict[str, str] = {
    "temp": "t",
    "u": "u",
    "v": "v",
    "ps": "sp",
    "tsurf": "skt",
    "co2ice": "co2ice",
    "dustcol": "tcdo",
}

# EMARS: variables are prefixed with ``anal_mean_``.
# After stripping the prefix, apply this rename map.
_RENAME_EMARS_SUFFIX: dict[str, str] = {
    "T": "t",
    "U": "u",
    "V": "v",
    "ps": "sp",
    "Surface_geopotential": "z_sfc",
}

# Auxiliary EMARS variables to drop (not spatial fields).
_EMARS_AUX_PREFIXES = (
    "anal_mean_Ls",
    "anal_mean_MY",
    "anal_mean_ak",
    "anal_mean_bk",
    "anal_mean_earth_",
    "anal_mean_emars_sol",
    "anal_mean_macda_sol",
    "anal_mean_mars_",
)

# Valid OpenMars era prefixes (order matters: first era fills first)
_OPENMARS_ERAS = ("MY24-27", "MY28-35")


def _merge_openmars_eras(ds: "xr.Dataset") -> "xr.Dataset":
    """Merge era-prefixed OpenMars variables into single continuous ones.

    ``MY24-27_temp`` and ``MY28-35_temp`` become ``t``, filling NaN
    gaps from one era with values from the other so the result is a
    continuous time series.
    """

    # Discover which base variables exist across eras
    base_vars: dict[str, list[str]] = {}  # suffix → [era-prefixed names]
    for var in list(ds.data_vars):
        for era in _OPENMARS_ERAS:
            prefix = f"{era}_"
            if var.startswith(prefix):
                suffix = var[len(prefix) :]
                base_vars.setdefault(suffix, []).append(var)
                break

    merged: dict[str, "xr.DataArray"] = {}
    drop: list[str] = []
    for suffix, era_vars in base_vars.items():
        std_name = _RENAME_OPENMARS_SUFFIX.get(suffix, suffix)
        # Start from the first era, fill NaNs with subsequent eras
        result = ds[era_vars[0]]
        for v in era_vars[1:]:
            result = result.fillna(ds[v])
        merged[std_name] = result
        drop.extend(era_vars)

    ds = ds.drop_vars(drop)
    for name, arr in merged.items():
        ds[name] = arr

    return ds


def _process_emars(ds: "xr.Dataset") -> "xr.Dataset":
    """Strip ``anal_mean_`` prefix, drop auxiliary variables, rename fields.

    EMARS variables are named ``anal_mean_T``, ``anal_mean_U``, etc.
    Auxiliary time/coordinate arrays (Earth dates, Ls, ak/bk, …) are
    dropped.  The ``pfull`` vertical dimension is renamed to ``level``
    with 1-indexed integers.  Extra lat/lon coordinates (``latu``,
    ``lonv``) are dropped.
    """
    # Drop auxiliary variables
    aux_drop = [v for v in ds.data_vars if any(str(v).startswith(p) for p in _EMARS_AUX_PREFIXES)]
    # Drop extra coordinates
    for coord in ("latu", "lonv", "phalf"):
        if coord in ds.coords:
            aux_drop.append(coord)
    if aux_drop:
        ds = ds.drop_vars(aux_drop)

    # Rename pfull → level with integer indices
    if "pfull" in ds.dims:
        n_lev = len(ds["pfull"])
        ds = ds.rename({"pfull": "level"})
        ds = ds.assign_coords(level=("level", np.arange(1, n_lev + 1)))

    # Strip anal_mean_ prefix and apply rename
    rename_map = {}
    for var in list(ds.data_vars):
        name = str(var)
        if name.startswith("anal_mean_"):
            suffix = name[len("anal_mean_") :]
            std = _RENAME_EMARS_SUFFIX.get(suffix, suffix)
            rename_map[name] = std
    if rename_map:
        ds = ds.rename(rename_map)

    return ds


def _sort_and_fill_gaps(ds: "xr.Dataset", raw_sols: np.ndarray, steps_per_sol: int) -> "xr.Dataset":
    """Sort the dataset by sol, deduplicate, and warn about gaps.

    Some datasets have their time axis out of order (MACDA interleaves
    Mars Year segments) and/or contain forward gaps (OpenMars MY27-28).
    This function:

    1. **Drops** time steps with NaN sol values.
    2. **Sorts** the dataset so sol values are monotonically increasing.
    3. **Removes duplicate** sol values (keeps first occurrence).
    4. **Warns** about gaps where consecutive sols jump by more than
       1.5x the expected step.  The gap boundaries should be listed
       in ``dates.missing`` in the recipe config so the pipeline
       excludes them from training.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with the original (sol-based) time coordinate.
    raw_sols : np.ndarray
        Raw sol values from the time coordinate.
    steps_per_sol : int
        Expected number of steps per sol (12 for MACDA/OpenMars,
        24 for EMARS).

    Returns
    -------
    xr.Dataset
        Sorted dataset with gap steps inserted (if any).
    """
    import xarray as xr

    sol_step = 1.0 / steps_per_sol
    n_orig = len(raw_sols)

    # ---- 1. Drop NaN sols ----
    valid_mask = ~np.isnan(raw_sols)
    if not np.all(valid_mask):
        n_nan = int((~valid_mask).sum())
        LOG.info("Dropping %d time steps with NaN sol values.", n_nan)
        valid_indices = np.where(valid_mask)[0]
        ds = ds.isel(time=valid_indices)
        raw_sols = raw_sols[valid_mask]

    # ---- 2. Sort by sol ----
    sort_order = np.argsort(raw_sols)
    if not np.array_equal(sort_order, np.arange(len(sort_order))):
        LOG.info("Sorting %d time steps by sol value.", len(raw_sols))
        ds = ds.isel(time=sort_order)
        raw_sols = raw_sols[sort_order]

    # ---- 3. Remove duplicate sols ----
    _, unique_idx = np.unique(raw_sols, return_index=True)
    if len(unique_idx) < len(raw_sols):
        n_dups = len(raw_sols) - len(unique_idx)
        LOG.info("Removing %d duplicate sol values.", n_dups)
        ds = ds.isel(time=unique_idx)
        raw_sols = raw_sols[unique_idx]

    # ---- 4. Detect gaps and insert NaN-filled steps ----
    # NaN insertion is essential: without it, the synthetic datetime→sol
    # mapping breaks for all steps after the gap, making forcings
    # (sol-of-day, sol-of-year, Ls, insolation) compute wrong values.
    # It also ensures that joined datasets share the same date axis.
    # The inserted NaN dates should be listed in ``dates.missing`` in
    # the recipe config so the pipeline skips them during training.

    diffs = np.diff(raw_sols)
    gap_indices = np.where(diffs > sol_step * 1.5)[0]

    if len(raw_sols) != n_orig:
        LOG.info("After sort/dedup: %d steps (was %d).", len(raw_sols), n_orig)

    if len(gap_indices) == 0:
        return ds

    pieces = []
    prev = 0
    total_inserted = 0
    for gi in gap_indices:
        n_missing = int(np.round((raw_sols[gi + 1] - raw_sols[gi]) / sol_step)) - 1
        if n_missing <= 0:
            continue

        LOG.warning(
            "Sol gap at step %d: sol %.2f -> %.2f (%d NaN steps inserted, ~%.0f sols). "
            "List these dates in 'dates.missing' to exclude from training.",
            gi + total_inserted,
            raw_sols[gi],
            raw_sols[gi + 1],
            n_missing,
            raw_sols[gi + 1] - raw_sols[gi],
        )

        insert_at = gi + 1
        pieces.append(ds.isel(time=slice(prev, insert_at)))

        # Build a NaN-filled dataset slice using dask arrays to avoid
        # materialising potentially huge arrays in memory (e.g. 6120
        # steps × 35 levels × 36 lat × 72 lon would be ~22 GB per var).
        import dask.array as da

        template = ds.isel(time=slice(insert_at - 1, insert_at))
        nan_data = {}
        for var in template.data_vars:
            spatial_shape = template[var].shape[1:]
            # Use same chunk sizes as the original data for the
            # spatial dimensions; single chunk for the time axis.
            spatial_chunks = tuple(s for s in spatial_shape)
            nan_arr = da.full(
                (n_missing,) + spatial_shape,
                np.nan,
                dtype=np.float32,
                chunks=(n_missing,) + spatial_chunks,
            )
            nan_data[var] = xr.DataArray(nan_arr, dims=template[var].dims)
        dummy_time = np.arange(n_missing, dtype="float64")
        nan_ds = xr.Dataset(nan_data, coords={"time": dummy_time})
        for coord in template.coords:
            if coord != "time" and coord in ds.coords:
                nan_ds = nan_ds.assign_coords({coord: ds.coords[coord]})
        pieces.append(nan_ds)

        prev = insert_at
        total_inserted += n_missing

    pieces.append(ds.isel(time=slice(prev, None)))
    ds = xr.concat(pieces, dim="time")
    LOG.info(
        "After sort + gap fill: %d total steps (%d NaN steps inserted).",
        len(ds.time),
        total_inserted,
    )
    return ds


_RENAME_CANONICAL: dict[str, dict[str, str]] = {
    CANONICAL_MACDA: _RENAME_MACDA,
    # OpenMars and EMARS are handled by dedicated functions.
}


def _postprocess_dataset(
    ds: "xr.Dataset",
    canonical: str,
    info: dict,
) -> "xr.Dataset":
    """Common post-processing shared by all backends.

    Handles: sol-gap fill, synthetic Earth-time grid assignment,
    dropping auxiliary vars, level renaming, and dataset-specific
    variable renaming/era merging.
    """
    freq_h = info.get("frequency_h", 2)
    steps_per_sol = info["steps_per_sol"]

    # ---- Detect sol gaps and insert NaN-filled steps ----
    if "time" in ds.coords:
        # .load() ensures we get a numpy array even if the coord is dask-backed
        raw_sols = np.asarray(ds["time"].load().values, dtype="float64")
        ds = _sort_and_fill_gaps(ds, raw_sols, steps_per_sol)

        n = len(ds["time"])
        grid_times = sols_to_regular_grid(n, frequency_h=freq_h)
        ds = ds.assign_coords(time=("time", grid_times))
        LOG.info(
            "Assigned %d-step synthetic %dh grid: %s .. %s",
            n,
            freq_h,
            grid_times[0],
            grid_times[-1],
        )

    # ---- Drop auxiliary variables that are not spatial fields ----
    to_drop = [v for v in _DROP_VARS if v in ds]
    if to_drop:
        ds = ds.drop_vars(to_drop)

    # ---- Rename 'lev' to 'level' and replace sigma values with ints ----
    # The default guesser recognizes name=="level" as a level coordinate.
    # We also replace the raw sigma values (0.9995, 0.9982, …) with
    # 1-indexed integers (1 = lowest / near-surface, N = top) so that
    # the downstream variable names read ``t_1`` instead of ``t_0.9994…``.
    if "lev" in ds.dims:
        n_levels = len(ds["lev"])
        ds = ds.rename({"lev": "level"})
        ds = ds.assign_coords(level=("level", np.arange(1, n_levels + 1)))

    # ---- Dataset-specific processing ----
    if canonical == CANONICAL_OPENMARS:
        ds = _merge_openmars_eras(ds)
        LOG.info("Merged OpenMars eras → variables: %s", list(ds.data_vars))
    elif canonical == CANONICAL_EMARS:
        ds = _process_emars(ds)
        LOG.info("Processed EMARS → variables: %s", list(ds.data_vars))
    else:
        rename_map = _RENAME_CANONICAL.get(canonical, {})
        rename_map = {k: v for k, v in rename_map.items() if k in ds}
        if rename_map:
            ds = ds.rename(rename_map)
            LOG.info("Renamed variables: %s", rename_map)

    return ds


def _open_hf_zarr(canonical: str) -> "xr.Dataset":
    """Open the unified Zarr v3 store from a HuggingFace ``datasets`` repo.

    Always opens the ``default`` (unified/combined) store listed in
    :data:`_KNOWN_STORES` for the given canonical Mars dataset.

    Parameters
    ----------
    canonical : str
        Canonical Mars dataset short name (e.g. ``"ARCO-MACDA"``).

    Returns
    -------
    xr.Dataset
        The opened dataset with time converted to Earth ``datetime64``.
    """
    import xarray as xr
    import zarr  # noqa: F401  – must be >= 3.0 for zarr-format-3 support
    from fsspec import filesystem

    # Silence the per-request HTTP logging from httpx / huggingface_hub
    for _logger_name in ("httpx", "huggingface_hub", "fsspec"):
        logging.getLogger(_logger_name).setLevel(logging.WARNING)

    info = _KNOWN_STORES[canonical]
    hf_info = info["hf"]
    store = hf_info["default"]
    repo = hf_info["repo"]

    hf_path = f"datasets/{repo}/{store}"
    LOG.info("Opening HuggingFace zarr store: %s", hf_path)

    fs = filesystem("hf", token=True)
    fs_map = fs.get_mapper(hf_path)

    ds = xr.open_zarr(
        fs_map,
        zarr_format=3,
        consolidated=False,
        decode_times=False,
    )

    return _postprocess_dataset(ds, canonical, info)


def _open_em_zarr(canonical: str) -> "xr.Dataset":
    """Open a Zarr store from the Earthmover / Arraylake catalogue.

    Uses :class:`arraylake.Client` to fetch a read-only session on the
    ``main`` branch of the repository, following the client pattern in
    https://github.com/GalacticBobster/ARCO-Mars-Examples.

    Group selection is fixed per dataset (matching what the HF backend
    exposes):

    * MACDA — single root group (``""``).
    * EMARS — ``mean`` (ensemble mean).
    * OpenMars — both era groups (``my24``, ``my28``) are opened and
      merged into a single continuous dataset, mirroring the HF
      unified store.

    Parameters
    ----------
    canonical : str
        Canonical Mars dataset short name.

    Returns
    -------
    xr.Dataset
        The opened (and post-processed) dataset with time converted
        to Earth ``datetime64``.
    """
    import xarray as xr
    from arraylake import Client

    info = _KNOWN_STORES[canonical]
    em_info = info["earthmover"]
    repo = em_info["repo"]

    client = Client()
    repo_handle = client.get_repo(repo)
    session = repo_handle.readonly_session("main")

    if canonical == CANONICAL_OPENMARS:
        # Merge the era groups into a single dataset with era-prefixed
        # variables so the downstream ``_merge_openmars_eras`` logic in
        # ``_postprocess_dataset`` can consume it unchanged (giving
        # parity with the HF unified store).
        era_to_prefix = {"my24": "MY24-27", "my28": "MY28-35"}
        era_groups = em_info["era_groups"]
        LOG.info(
            "Opening Earthmover OpenMars groups: %s (repo %s)",
            era_groups,
            repo,
        )
        pieces: list[xr.Dataset] = []
        for eg in era_groups:
            ds_era = xr.open_zarr(
                session.store,
                group=eg,
                consolidated=False,
                decode_times=False,
            )
            prefix = era_to_prefix.get(eg, eg)
            rename = {v: f"{prefix}_{v}" for v in ds_era.data_vars}
            pieces.append(ds_era.rename(rename))
        # Outer merge on time so eras with disjoint sol ranges become
        # a single dataset with NaN where each era has no data.
        ds = xr.merge(pieces, join="outer")
    else:
        grp = em_info.get("default_group", "")
        LOG.info("Opening Earthmover zarr: repo=%s group=%r", repo, grp)
        ds = xr.open_zarr(
            session.store,
            group=grp,
            consolidated=False,
            decode_times=False,
        )

    return _postprocess_dataset(ds, canonical, info)


def _open_arco_zarr(dataset: str, backend: str = "hf") -> "xr.Dataset":
    """Backend-dispatching opener for ARCO-Mars datasets.

    Parameters
    ----------
    dataset : str
        Any accepted dataset id (short name or fully-qualified for
        either backend); see :func:`_normalise_dataset`.
    backend : {"hf", "earthmover"}
        Which hosting backend to use.  ``"hf"`` uses the HuggingFace
        Datasets Zarr stores; ``"earthmover"`` uses the Arraylake
        catalogue.

    Returns
    -------
    xr.Dataset
        The opened (and post-processed) dataset with time converted
        to Earth ``datetime64``.
    """
    canonical = _normalise_dataset(dataset)
    if backend == "hf":
        return _open_hf_zarr(canonical)
    if backend == "earthmover":
        return _open_em_zarr(canonical)
    raise ValueError(f"Unknown backend '{backend}'.  Expected 'hf' or 'earthmover'.")


# ---------------------------------------------------------------------------
# Source class
# ---------------------------------------------------------------------------
class ArcoMarsSource(XarraySourceBase):
    """Anemoi-datasets source for ARCO Mars reanalysis.

    Supports multiple underlying datasets (MACDA, OpenMars, EMARS) —
    all published as ARCO Zarr stores — served from either of two
    hosting backends:

    * ``backend: hf`` (default) — HuggingFace Datasets repos
      (``ananyo01/ARCO-*``), opened via ``fsspec``.
    * ``backend: earthmover`` — Earthmover / Arraylake catalogue
      (``arco-planetary/ARCO-*``), opened via ``arraylake.Client``.

    In both cases the native Mars-sol time axis is converted to a
    synthetic Earth ``datetime64`` grid and the result is exposed
    through the standard anemoi xarray field-list machinery.

    Parameters
    ----------
    context : Any
        Pipeline context.
    dataset : str
        Mars dataset id. Accepts a canonical short name (``ARCO-MACDA``,
        ``ARCO-OpenMars``, ``ARCO-EMARS``), an HF-qualified id
        (``ananyo01/ARCO-MACDA``), or an Earthmover-qualified id
        (``arco-planetary/ARCO-MACDA``). The canonical name is derived
        automatically via :func:`_normalise_dataset` and used to look up
        the correct repository for the selected ``backend``.
    args : Any
        Additional positional arguments passed to the xarray field-list loader.
    backend : str, optional
        Hosting backend, one of ``"hf"`` (default) or ``"earthmover"``.
    kwargs : Any
        Additional keyword arguments passed to the xarray field-list loader.
    """

    emoji = "🔴"  # Mars!

    def __init__(
        self,
        context: Any,
        dataset: str,
        *args: Any,
        backend: str = "hf",
        **kwargs: Any,
    ) -> None:
        self.flavour = kwargs.pop("flavour", None)
        self.patch = kwargs.pop("patch", None)

        super().__init__(context, **kwargs)

        self._dataset = dataset
        self._canonical = _normalise_dataset(dataset)
        self._backend = backend
        self._ds: xr.Dataset | None = None  # lazy

    # -- lazy open so the heavy I/O only happens at execution time ----------
    def _get_dataset(self) -> "xr.Dataset":
        if self._ds is None:
            self._ds = _open_arco_zarr(
                self._canonical,
                backend=self._backend,
            )
        return self._ds

    def execute_valid_dates(self, dates: DateList) -> Any:
        """Load fields for the requested Earth dates.

        The *dates* coming from the pipeline are Earth ``datetime``
        objects.  We pre-load the requested time slice into memory
        so the remote zarr chunks are fetched once (not per-field).
        """
        import pandas as pd

        ds = self._get_dataset()

        # Convert dates to datetime64 for xarray selection
        dt_dates = pd.DatetimeIndex([np.datetime64(d, "ns") for d in dates])

        # Batch-select the full date range and load into memory.
        # This fetches each zarr chunk at most once instead of once
        # per (date, variable, level) combination.
        ds_slice = ds.sel(time=dt_dates).compute()

        return load_one(
            self.emoji,
            self.context,
            [d.isoformat() for d in dates],
            ds_slice,
            flavour=self.flavour,
            patch=self.patch,
            **self.kwargs,
        )
