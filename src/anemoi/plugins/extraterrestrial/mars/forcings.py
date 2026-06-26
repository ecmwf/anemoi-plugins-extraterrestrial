# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Mars-specific forcings source for anemoi-datasets (gridded).

Drop-in replacement for the Earth ``forcings`` source that computes
Mars-appropriate values for the same parameter names.  The synthetic
2h Earth-datetime grid is converted back to Mars sols so that the
daily and yearly cycles reflect Mars, not Earth, periodicity.

Supported parameters
--------------------
=================  ========================================================
Parameter          Description
=================  ========================================================
cos_latitude       cos(latitude)  — same as Earth
sin_latitude       sin(latitude)  — same as Earth
cos_longitude      cos(longitude) — same as Earth
sin_longitude      sin(longitude) — same as Earth
cos_julian_day     cos(sol-of-year angle), period = 1 Mars year (~668.6 sols)
sin_julian_day     sin(sol-of-year angle)
cos_local_time     cos(local-sol-of-day angle), period = 1 Mars sol
sin_local_time     sin(local-sol-of-day angle)
insolation         cos(Mars solar zenith angle), clipped to [0, 1]
solar_longitude    areocentric solar longitude Ls in [0, 360) degrees
=================  ========================================================

Note: for topography (surface altitude), include the static field
directly from the dataset rather than computing it as a forcing.
EMARS provides ``z_sfc`` (Surface_geopotential); MACDA provides
``z`` at level 1 (near-surface geopotential).  See Roy et al. (2026)
§4.3.1 for the GraphCast forcing design.

Example YAML
------------
::

    input:
      join:
        - openmars:
            dataset: ananyo01/ARCO-MACDA
            param: [sp, skt]
        - mars_forcings:
            dataset: ananyo01/ARCO-MACDA
            param:
              - cos_latitude
              - sin_latitude
              - cos_julian_day
              - sin_julian_day
              - cos_local_time
              - sin_local_time
              - solar_longitude
              - insolation
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.types import DateList

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mars orbital / rotational constants
# ---------------------------------------------------------------------------
# Tropical year in sols (Ls = 0 → Ls = 360)
MARS_YEAR_SOLS = 668.5991

# Mars obliquity (axial tilt) in radians
_MARS_OBLIQUITY_RAD = np.deg2rad(25.19)

# Grid epoch — must match the value in source.py
_GRID_EPOCH = dt.datetime(2000, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def _datetime_to_sol(
    date: dt.datetime,
    steps_per_sol: int,
    frequency_h: int,
) -> float:
    """Convert a synthetic Earth datetime back to a Mars sol number.

    Sol 0 corresponds to the grid epoch (the first time step in any
    dataset opened by the openmars source).
    """
    delta = date - _GRID_EPOCH
    step_index = delta.total_seconds() / (frequency_h * 3600)
    return step_index / steps_per_sol


def _sol_of_year(sol: float) -> float:
    """Fractional position within the Mars year, in [0, MARS_YEAR_SOLS)."""
    return sol % MARS_YEAR_SOLS


def _sol_of_day(sol: float) -> float:
    """Fractional position within the Mars sol, in [0, 1)."""
    return sol % 1.0


def _solar_longitude_rad(sol: float) -> float:
    """Approximate solar longitude Ls in radians from sol number.

    This is a simple linear approximation: Ls = sol / MARS_YEAR_SOLS * 2pi.
    Accurate enough for forcings (the model learns the phase).
    """
    return _sol_of_year(sol) / MARS_YEAR_SOLS * 2.0 * np.pi


def _mars_solar_declination(ls_rad: float) -> float:
    """Mars solar declination angle in radians.

    dec = arcsin(sin(obliquity) * sin(Ls))
    """
    return np.arcsin(np.sin(_MARS_OBLIQUITY_RAD) * np.sin(ls_rad))


# ---------------------------------------------------------------------------
# Forcing computation
# ---------------------------------------------------------------------------
class MarsForcingMaker:
    """Computes Mars-specific forcing fields on a lat/lon grid."""

    SUPPORTED = {
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "insolation",
        "solar_longitude",
    }

    def __init__(
        self,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        steps_per_sol: int,
        frequency_h: int,
    ) -> None:
        self.lat_rad = np.deg2rad(latitudes)
        self.lon_deg = longitudes
        self.n_points = len(latitudes)
        self.steps_per_sol = steps_per_sol
        self.frequency_h = frequency_h

    # -- spatial (time-independent) -----------------------------------------
    def cos_latitude(self, date: dt.datetime) -> np.ndarray:
        return np.cos(self.lat_rad)

    def sin_latitude(self, date: dt.datetime) -> np.ndarray:
        return np.sin(self.lat_rad)

    def cos_longitude(self, date: dt.datetime) -> np.ndarray:
        return np.cos(np.deg2rad(self.lon_deg))

    def sin_longitude(self, date: dt.datetime) -> np.ndarray:
        return np.sin(np.deg2rad(self.lon_deg))

    # -- sol-of-year (Mars yearly cycle) ------------------------------------
    def cos_julian_day(self, date: dt.datetime) -> np.ndarray:
        sol = _datetime_to_sol(date, self.steps_per_sol, self.frequency_h)
        angle = _sol_of_year(sol) / MARS_YEAR_SOLS * 2.0 * np.pi
        return np.full(self.n_points, np.cos(angle))

    def sin_julian_day(self, date: dt.datetime) -> np.ndarray:
        sol = _datetime_to_sol(date, self.steps_per_sol, self.frequency_h)
        angle = _sol_of_year(sol) / MARS_YEAR_SOLS * 2.0 * np.pi
        return np.full(self.n_points, np.sin(angle))

    # -- solar longitude Ls (Mars seasonal coordinate) ----------------------
    def solar_longitude(self, date: dt.datetime) -> np.ndarray:
        """Areocentric solar longitude Ls in [0, 360) degrees.

        Used by Mars-Adapted GraphCast (Roy et al., 2026) as a direct
        forcing variable encoding martian seasonality.  Ls = 0 at
        northern spring equinox, 90 at summer solstice, 180 at autumn
        equinox, 270 at winter solstice.

        Note: this is a linear approximation; Mars's eccentric orbit
        causes Ls to advance non-uniformly, but the model can learn
        the residual from the prognostic fields.
        """
        sol = _datetime_to_sol(date, self.steps_per_sol, self.frequency_h)
        ls_deg = (_sol_of_year(sol) / MARS_YEAR_SOLS * 360.0) % 360.0
        return np.full(self.n_points, ls_deg)

    # -- sol-of-day (Mars daily cycle, local) -------------------------------
    def cos_local_time(self, date: dt.datetime) -> np.ndarray:
        sol = _datetime_to_sol(date, self.steps_per_sol, self.frequency_h)
        frac = _sol_of_day(sol)
        # local time offset by longitude (360° = 1 sol)
        local_frac = (frac + self.lon_deg / 360.0) % 1.0
        angle = local_frac * 2.0 * np.pi
        return np.cos(angle)

    def sin_local_time(self, date: dt.datetime) -> np.ndarray:
        sol = _datetime_to_sol(date, self.steps_per_sol, self.frequency_h)
        frac = _sol_of_day(sol)
        local_frac = (frac + self.lon_deg / 360.0) % 1.0
        angle = local_frac * 2.0 * np.pi
        return np.sin(angle)

    # -- insolation (Mars cos solar zenith angle) ---------------------------
    def insolation(self, date: dt.datetime) -> np.ndarray:
        sol = _datetime_to_sol(date, self.steps_per_sol, self.frequency_h)
        ls_rad = _solar_longitude_rad(sol)
        dec = _mars_solar_declination(ls_rad)

        frac = _sol_of_day(sol)
        local_frac = (frac + self.lon_deg / 360.0) % 1.0
        hour_angle = (local_frac - 0.5) * 2.0 * np.pi

        cos_sza = np.sin(self.lat_rad) * np.sin(dec) + np.cos(self.lat_rad) * np.cos(dec) * np.cos(hour_angle)
        return np.clip(cos_sza, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
class MarsForcingsSource(Source):
    """Anemoi-datasets source producing Mars-specific forcings on a grid.

    Uses a ``template`` + ``param`` interface similar to the Earth
    ``forcings`` source.  The *template* is another source in the
    recipe whose grid (lat/lon) is reused.

    Parameters
    ----------
    context : Any
        Pipeline context.
    dataset : str
        HuggingFace dataset id (e.g. ``"ananyo01/ARCO-MACDA"``).
        The grid and temporal resolution are looked up automatically.
    param : list[str]
        Which forcing parameters to compute.
    """

    emoji = "🔴"

    def __init__(
        self,
        context: Any,
        dataset: str,
        param: list[str],
    ) -> None:
        """Initialise Mars forcings source.

        Parameters
        ----------
        context : Any
            Pipeline context.
        dataset : str
            HuggingFace dataset id (e.g. ``"ananyo01/ARCO-MACDA"``).
        param : list[str]
            Which forcing parameters to compute.
        """
        super().__init__(context)
        self.param = param if isinstance(param, list) else [param]

        from anemoi.plugins.extraterrestrial.mars.source import _KNOWN_STORES

        info = _KNOWN_STORES.get(dataset)
        if info is None:
            raise ValueError(f"Unknown dataset '{dataset}' for mars_forcings.  " f"Known: {list(_KNOWN_STORES)}")
        self._nlat = info["nlat"]
        self._nlon = info["nlon"]
        self._steps_per_sol = info["steps_per_sol"]
        self._frequency_h = info["frequency_h"]

        unknown = set(self.param) - MarsForcingMaker.SUPPORTED
        if unknown:
            raise ValueError(
                f"Unknown mars_forcings param(s): {unknown}.  " f"Supported: {sorted(MarsForcingMaker.SUPPORTED)}"
            )

    def execute_valid_dates(self, dates: DateList) -> Any:
        from earthkit.data import from_source

        self.context.trace(self.emoji, f"mars_forcings({self.param})")

        # Build grid from the catalogue metadata
        lat_spacing = 180.0 / self._nlat
        lon_spacing = 360.0 / self._nlon
        lats = np.arange(90.0 - lat_spacing / 2, -90.0, -lat_spacing)
        lons = np.arange(-180.0, 180.0, lon_spacing)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        flat_lats = lat_grid.ravel()
        flat_lons = lon_grid.ravel()

        # Use earthkit to create a template field with this grid
        template_fields = from_source(
            "forcings",
            latitudes=flat_lats,
            longitudes=flat_lons,
            date=[dates[0]],
            param=["cos_latitude"],
        )
        field0 = template_fields[0]

        maker = MarsForcingMaker(
            flat_lats,
            flat_lons,
            steps_per_sol=self._steps_per_sol,
            frequency_h=self._frequency_h,
        )

        from anemoi.transform.fields import new_field_with_metadata
        from anemoi.transform.fields import new_fieldlist_from_list

        result = []
        for date in dates:
            if not isinstance(date, dt.datetime):
                date = dt.datetime.fromisoformat(str(date))
            for param in self.param:
                values = getattr(maker, param)(date)
                result.append(
                    new_field_with_metadata(
                        field0,
                        values=values,
                        param=param,
                        variable=param,
                        valid_datetime=date.isoformat(),
                        units="dimensionless",
                    )
                )

        return new_fieldlist_from_list(result)
