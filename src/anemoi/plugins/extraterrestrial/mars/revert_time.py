# SPDX-FileCopyrightText: 2026 Anemoi contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Revert the synthetic Earth-datetime grid in NetCDF outputs back to Mars sols.

The anemoi pipeline requires a regular Earth-time grid, so the
:mod:`~anemoi.plugins.extraterrestrial.mars.source` module replaces the
native Mars-sol time axis with a synthetic grid starting at
``2000-01-01T00:00:00`` where **24 synthetic Earth-hours = 1 Mars sol**.

NetCDF files produced by inference (e.g. ``output.nc``) carry this
synthetic time regardless of their temporal resolution (2h, 6h, …).
The conversion back is trivial:

    sol = (time − epoch) in days

This module provides utilities to revert that conversion:

* Replace the synthetic ``datetime64`` time coordinate with fractional
  Mars sols (float64).
* Optionally add an ``Ls`` (areocentric solar longitude, 0–360°)
  coordinate.
* Optionally add a ``mars_year`` coordinate indicating the Mars Year
  number (MY24-based counting from sol 0).

No configuration is needed — the tool reads the time axis and converts.

Usage as a library
------------------
::

    from anemoi.plugins.extraterrestrial.mars.revert_time import revert_nc_time

    revert_nc_time("output.nc", add_ls=True, add_mars_year=True)

    # Or on an in-memory Dataset:
    from anemoi.plugins.extraterrestrial.mars.revert_time import revert_time_axis

    ds = revert_time_axis(ds)

Usage from the command line
---------------------------
::

    mars-revert-time output.nc
    mars-revert-time output.nc --add-ls --add-mars-year
    mars-revert-time output.nc -o output_mars.nc --add-ls
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must stay in sync with source.py / forcings.py
# ---------------------------------------------------------------------------
_GRID_EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")

# Tropical Mars year in sols (Ls 0 → Ls 360)
MARS_YEAR_SOLS = 668.5991

# Nanoseconds in one day (= one sol on the synthetic grid)
_NS_PER_DAY = 86_400_000_000_000


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------
def datetime64_to_sol(times: np.ndarray) -> np.ndarray:
    """Convert synthetic Earth ``datetime64`` values to Mars sol numbers.

    On the synthetic grid 24 Earth-hours = 1 Mars sol, so the
    conversion is simply the elapsed time since the grid epoch
    measured in days.

    Parameters
    ----------
    times : np.ndarray
        Array of ``numpy.datetime64`` values on the synthetic grid.

    Returns
    -------
    np.ndarray
        Array of float64 Mars sol values.  Sol 0 corresponds to the
        grid epoch (``2000-01-01T00:00:00``).
    """
    times_ns = np.asarray(times, dtype="datetime64[ns]")
    delta_ns = (times_ns - _GRID_EPOCH).astype("float64")
    return delta_ns / _NS_PER_DAY


def sol_to_solar_longitude(sols: np.ndarray) -> np.ndarray:
    """Compute approximate areocentric solar longitude Ls from sol values.

    Uses the same linear approximation as
    :func:`~anemoi.plugins.extraterrestrial.mars.forcings._solar_longitude_rad`,
    returned in degrees [0, 360).

    Parameters
    ----------
    sols : np.ndarray
        Array of float64 Mars sol values.

    Returns
    -------
    np.ndarray
        Ls in degrees, in the range [0, 360).
    """
    sol_of_year = np.mod(sols, MARS_YEAR_SOLS)
    return np.mod(sol_of_year / MARS_YEAR_SOLS * 360.0, 360.0)


def sol_to_mars_year(sols: np.ndarray, base_my: int = 24) -> np.ndarray:
    """Compute Mars Year number from sol values.

    Sol 0 in the synthetic grid corresponds to Mars Year *base_my*
    Ls = 0 (the datasets start at MY24 northern spring equinox).

    Parameters
    ----------
    sols : np.ndarray
        Array of float64 Mars sol values.
    base_my : int
        Mars Year number corresponding to sol 0.  Default is 24
        (MY24 Ls = 0 for MACDA / OpenMars).

    Returns
    -------
    np.ndarray
        Integer Mars Year numbers (int32).
    """
    return (np.floor(sols / MARS_YEAR_SOLS) + base_my).astype(np.int32)


# ---------------------------------------------------------------------------
# In-memory Dataset reversion
# ---------------------------------------------------------------------------
def revert_time_axis(
    ds: "xr.Dataset",
    add_ls: bool = False,
    add_mars_year: bool = False,
    base_my: int = 24,
) -> "xr.Dataset":
    """Revert the time axis of an xarray Dataset from synthetic Earth datetimes to Mars sols.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``time`` coordinate on the synthetic Earth grid.
    add_ls : bool
        If ``True``, add an ``Ls`` coordinate (solar longitude in
        degrees, 0-360).
    add_mars_year : bool
        If ``True``, add a ``mars_year`` coordinate (integer MY
        number).
    base_my : int
        Mars Year number corresponding to sol 0.

    Returns
    -------
    xr.Dataset
        Dataset with the ``time`` coordinate replaced by Mars sols
        (renamed to ``sol``), plus optional ``Ls`` and ``mars_year``
        coordinates.

    Raises
    ------
    ValueError
        If the dataset has no ``time`` dimension.
    """
    if "time" not in ds.dims:
        raise ValueError(
            "Dataset has no 'time' dimension.  Cannot revert time axis.  " f"Available dimensions: {list(ds.dims)}"
        )

    sols = datetime64_to_sol(ds["time"].values)

    LOG.info(
        "Reverting %d time steps: sol %.4f .. %.4f",
        len(sols),
        sols[0],
        sols[-1],
    )

    # Replace the time coordinate with sols
    ds = ds.assign_coords(time=("time", sols))
    ds = ds.rename({"time": "sol"})

    # Update attributes for CF compliance
    ds["sol"].attrs = {
        "long_name": "Mars solar day (sol) since MY24 Ls=0",
        "units": "sols",
        "calendar": "mars",
    }

    # ---- Optional: solar longitude Ls ----
    if add_ls:
        ls_vals = sol_to_solar_longitude(sols)
        ds = ds.assign_coords(Ls=("sol", ls_vals))
        ds["Ls"].attrs = {
            "long_name": "Areocentric solar longitude",
            "units": "degrees",
            "valid_range": [0.0, 360.0],
        }
        LOG.info("Added Ls coordinate: %.2f° .. %.2f°", ls_vals[0], ls_vals[-1])

    # ---- Optional: Mars year ----
    if add_mars_year:
        my_vals = sol_to_mars_year(sols, base_my=base_my)
        ds = ds.assign_coords(mars_year=("sol", my_vals))
        ds["mars_year"].attrs = {
            "long_name": "Mars Year number",
            "units": "1",
        }
        LOG.info(
            "Added mars_year coordinate: MY%d .. MY%d",
            my_vals[0],
            my_vals[-1],
        )

    return ds


# ---------------------------------------------------------------------------
# NetCDF file-level reversion
# ---------------------------------------------------------------------------
def revert_nc_time(
    input_path: str | Path,
    output_path: str | Path | None = None,
    add_ls: bool = False,
    add_mars_year: bool = False,
    base_my: int = 24,
    overwrite: bool = False,
) -> "xr.Dataset":
    """Revert the synthetic Earth-time axis in a NetCDF file to Mars sols.

    Opens a NetCDF file (e.g. ``output.nc`` produced by anemoi
    inference), replaces its ``time`` coordinate with fractional Mars
    sols, and optionally adds ``Ls`` and ``mars_year`` coordinates.

    Parameters
    ----------
    input_path : str or Path
        Path to the input NetCDF file.
    output_path : str, Path, or None
        Where to write the reverted file.  If ``None``, a path is
        generated by inserting ``_mars`` before the ``.nc`` extension.
    add_ls : bool
        If ``True``, add an ``Ls`` coordinate (solar longitude in
        degrees, 0–360).
    add_mars_year : bool
        If ``True``, add a ``mars_year`` coordinate (integer MY
        number).
    base_my : int
        Mars Year number corresponding to sol 0. Default 24.
    overwrite : bool
        If ``True``, allow overwriting the output file.

    Returns
    -------
    xr.Dataset
        The dataset with the reverted time axis (also saved to disk).

    Raises
    ------
    ValueError
        If the file has no ``time`` dimension.
    FileExistsError
        If *output_path* exists and *overwrite* is ``False``.
    """
    import xarray as xr

    input_path = Path(input_path)

    LOG.info("Opening %s", input_path)
    ds = xr.open_dataset(input_path)

    ds = revert_time_axis(
        ds,
        add_ls=add_ls,
        add_mars_year=add_mars_year,
        base_my=base_my,
    )

    # ---- Write output ----
    if output_path is None:
        stem = input_path.stem
        output_path = input_path.with_name(f"{stem}_mars.nc")
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists.  " f"Use overwrite=True or choose a different path."
        )

    LOG.info("Writing reverted dataset to %s", output_path)
    ds.to_netcdf(output_path)
    LOG.info("Done.  %d time steps reverted.", len(ds.sol))

    return ds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Command-line interface for reverting NetCDF time axes."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mars-revert-time",
        description=(
            "Revert the synthetic Earth-datetime grid in a NetCDF output "
            "file back to Mars sols.  No configuration needed — the tool "
            "reads the time axis and converts (24 Earth-hours = 1 sol).  "
            "Optionally adds solar longitude (Ls) and Mars Year coordinates."
        ),
        epilog=(
            "Examples:\n"
            "  mars-revert-time output.nc\n"
            "  mars-revert-time output.nc --add-ls --add-mars-year\n"
            "  mars-revert-time output.nc -o output_mars.nc --add-ls\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input NetCDF file (e.g. output.nc).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=("Output NetCDF path.  Defaults to <input>_mars.nc " "(e.g. output.nc -> output_mars.nc)."),
    )
    parser.add_argument(
        "--add-ls",
        action="store_true",
        help="Add an Ls (solar longitude) coordinate.",
    )
    parser.add_argument(
        "--add-mars-year",
        action="store_true",
        help="Add a mars_year coordinate.",
    )
    parser.add_argument(
        "--base-my",
        type=int,
        default=24,
        help="Mars Year number for sol 0 (default: 24).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ds = revert_nc_time(
        input_path=args.input,
        output_path=args.output,
        add_ls=args.add_ls,
        add_mars_year=args.add_mars_year,
        base_my=args.base_my,
        overwrite=args.overwrite,
    )

    # Print summary
    print(f"\nReverted {len(ds.sol)} time steps")
    print(f"  Sol range: {float(ds.sol[0]):.4f} .. {float(ds.sol[-1]):.4f}")
    if "Ls" in ds.coords:
        print(f"  Ls range:  {float(ds.Ls[0]):.2f}° .. {float(ds.Ls[-1]):.2f}°")
    if "mars_year" in ds.coords:
        print(f"  MY range:  MY{int(ds.mars_year[0])} .. MY{int(ds.mars_year[-1])}")
    print(f"  Variables: {list(ds.data_vars)}")


if __name__ == "__main__":
    main()
