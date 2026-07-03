# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# type: ignore[reportPossiblyUnboundVariable]

"""Dynamic distance forcing between arbitrary solar system bodies.

Provides a :class:`DistanceForcingMaker` that resolves attribute names
of the form ``distance_from_{origin}_to_{target}`` at runtime, using
``__getattr__`` to parse the body names and dispatch to astropy for
the actual ephemeris calculation.

Three variants are supported for each body pair:

``distance_from_{origin}_to_{target}``
    Per-observer distance from *origin* body surface locations (given by
    latitudes/longitudes on that body) to the *target* body centre, in km.
    Returns an array of shape ``(N,)`` where *N* is the number of grid
    points.

``singular_distance_from_{origin}_to_{target}``
    Scalar centre-to-centre distance (no observer grid), in km.
    Returns a single float.

``delta_distance_from_{origin}_to_{target}``
    Per-observer distance minus the instantaneous maximum, giving a
    non-positive array that encodes tidal-like spatial variation without
    the large offset.  Returns an array of shape ``(N,)``.

Body names are matched case-insensitively against astropy's solar
system body catalogue (sun, mercury, venus, earth, moon, mars, jupiter,
saturn, uranus, neptune, pluto) and the JPL kernel extensions (phobos,
deimos, io, europa, ganymede, callisto, titan, triton, charon, etc.).

Example usage
-------------
::

    maker = DistanceForcingMaker()

    # Attribute-style access -- body pair parsed from the name
    d = maker.distance_from_earth_to_moon(date, latitudes, longitudes)
    s = maker.singular_distance_from_earth_to_moon(date, latitudes, longitudes)
    dd = maker.delta_distance_from_earth_to_moon(date, latitudes, longitudes)

    # Works for any body pair
    d2 = maker.distance_from_mars_to_phobos(date, latitudes, longitudes)

    # Or call the core functions directly
    d3 = compute_distance("mars", "phobos", date, latitudes, longitudes)
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

import numpy as np
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.types import DateList

LOG = logging.getLogger(__name__)

ASTROPY_AVAILABLE = True
try:
    import astropy.units as u
    from astropy.coordinates import get_body
    from astropy.coordinates import solar_system_ephemeris
    from astropy.time import Time
except ImportError:
    ASTROPY_AVAILABLE = False

# Regex patterns for the three attribute name variants.
# Captures (origin, target) from names like:
#   distance_from_earth_to_moon
#   singular_distance_from_mars_to_phobos
#   delta_distance_from_earth_to_moon
_RE_DISTANCE = re.compile(r"^distance_from_([a-z][a-z0-9_]*)_to_([a-z][a-z0-9_]*)$")
_RE_SINGULAR = re.compile(r"^singular_distance_from_([a-z][a-z0-9_]*)_to_([a-z][a-z0-9_]*)$")
_RE_DELTA = re.compile(r"^delta_distance_from_([a-z][a-z0-9_]*)_to_([a-z][a-z0-9_]*)$")

# Bodies supported by astropy's built-in ephemeris (``builtin``).
# The ``jpl`` ephemeris extends this with Pluto but not satellite moons.
# We try ``jpl`` first, falling back to ``builtin``.
_BUILTIN_BODIES = frozenset(
    {
        "sun",
        "mercury",
        "venus",
        "earth",
        "moon",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    }
)

# Bodies available via the JPL planetary ephemeris (DE440/DE441).
# Note: these kernels cover planetary *system* barycentres, Earth's
# Moon, and Pluto.  They do NOT include satellite moons (Phobos,
# Deimos, Io, Europa, etc.).
_JPL_BODIES = frozenset({"pluto"})

# Satellite moons not in any standard ephemeris.  These are resolved
# via a circular-orbit approximation around their parent body.
# Tuple: (parent_body, semi_major_axis_km, orbital_period_days)
# Sources: NASA planetary fact sheets.
_SATELLITE_ORBITS: dict[str, tuple[str, float, float]] = {
    "phobos": ("mars", 9376.0, 0.31891),
    "deimos": ("mars", 23463.2, 1.26244),
    "io": ("jupiter", 421700.0, 1.769),
    "europa": ("jupiter", 671034.0, 3.551),
    "ganymede": ("jupiter", 1070412.0, 7.155),
    "callisto": ("jupiter", 1882709.0, 16.689),
    "titan": ("saturn", 1221870.0, 15.945),
    "triton": ("neptune", 354759.0, 5.877),
    "charon": ("pluto", 19591.4, 6.387),
}

# All recognised body names.
KNOWN_BODIES = _BUILTIN_BODIES | _JPL_BODIES | frozenset(_SATELLITE_ORBITS.keys())

# ---------------------------------------------------------------------------
# Mean volumetric radii in km.
# Sources: NASA planetary fact sheets, JPL small-body database.
# Used to place observers on a body's surface via spherical coordinates.
# ---------------------------------------------------------------------------
_BODY_RADIUS_KM: dict[str, float] = {
    "sun": 695700.0,
    "mercury": 2439.7,
    "venus": 6051.8,
    "earth": 6371.0,
    "moon": 1737.4,
    "mars": 3389.5,
    "jupiter": 69911.0,
    "saturn": 58232.0,
    "uranus": 25362.0,
    "neptune": 24622.0,
    "pluto": 1188.3,
    "phobos": 11.27,
    "deimos": 6.2,
    "io": 1821.6,
    "europa": 1560.8,
    "ganymede": 2634.1,
    "callisto": 2410.3,
    "titan": 2574.7,
    "triton": 1353.4,
    "charon": 606.0,
}

# ---------------------------------------------------------------------------
# Sidereal rotation periods in days.
# Used to rotate body-fixed observer positions into the inertial frame.
# Sources: NASA planetary fact sheets.
# Negative values indicate retrograde rotation (Venus, Uranus, Pluto).
# Tidally locked moons use their orbital period.
# ---------------------------------------------------------------------------
_ROTATION_PERIOD_DAYS: dict[str, float] = {
    "sun": 25.38,
    "mercury": 58.646,
    "venus": -243.025,
    "earth": 0.99726968,
    "moon": 27.322,  # tidally locked
    "mars": 1.02595675,
    "jupiter": 0.41354,
    "saturn": 0.44401,
    "uranus": -0.71833,
    "neptune": 0.67125,
    "pluto": -6.38723,
    "phobos": 0.31891,  # tidally locked
    "deimos": 1.26244,  # tidally locked
    "io": 1.769,  # tidally locked
    "europa": 3.551,  # tidally locked
    "ganymede": 7.155,  # tidally locked
    "callisto": 16.689,  # tidally locked
    "titan": 15.945,  # tidally locked
    "triton": 5.877,  # tidally locked (retrograde orbit)
    "charon": 6.387,  # tidally locked
}


def _require_astropy() -> None:
    """Raise ImportError if astropy is not installed."""
    if not ASTROPY_AVAILABLE:
        raise ImportError(
            "`astropy` is required for distance forcing calculations. "
            "Install with: pip install 'anemoi-plugins-extraterrestrial[distances]'"
        )


def _resolve_body(name: str) -> str:
    """Validate and normalise a body name for astropy.

    Multi-word body names in attributes use underscores (e.g.
    ``new_horizons``), but astropy expects plain lowercase strings
    for standard bodies.  We strip underscores and match against
    the known set.

    Parameters
    ----------
    name : str
        Body name from the parsed attribute, e.g. ``"earth"``,
        ``"phobos"``.

    Returns
    -------
    str
        Normalised body name suitable for ``astropy.coordinates.get_body``.

    Raises
    ------
    ValueError
        If the body name is not recognised.
    """
    canonical = name.lower().replace("_", " ").strip()
    # Most bodies are single words; just use the lowercase form.
    if canonical.replace(" ", "") in {b.replace(" ", "") for b in KNOWN_BODIES}:
        # Return the form astropy expects (space-separated for
        # multi-word names, though none of the standard bodies have
        # spaces -- kept for forward compatibility).
        return canonical
    raise ValueError(f"Unknown solar system body '{name}'.  Known bodies: {sorted(KNOWN_BODIES)}")


def _get_body_xyz(body: str, time: "Time") -> np.ndarray:
    """Get the GCRS Cartesian position of *body* at *time* in km.

    For bodies in the standard ephemerides (planets, Moon, Pluto),
    uses astropy's ``get_body`` directly.  For satellite moons
    (Phobos, Io, Titan, etc.) that are not in DE440/DE441, computes
    the parent body's position and adds a circular-orbit offset.

    Parameters
    ----------
    body : str
        Normalised body name.
    time : astropy.time.Time
        Observation time.

    Returns
    -------
    np.ndarray
        Shape ``(3,)`` array of [x, y, z] in km (GCRS frame).
    """
    # Satellite moons: parent position + circular orbit offset.
    if body in _SATELLITE_ORBITS:
        parent, semi_major_km, period_days = _SATELLITE_ORBITS[body]
        parent_xyz = _get_body_xyz(parent, time)

        # Mean anomaly from J2000.0 epoch (arbitrary but consistent).
        elapsed_days = (time - Time("J2000.0")).jd
        angle = 2.0 * np.pi * (elapsed_days / period_days)

        # Circular orbit in the ecliptic plane (adequate for forcings
        # -- the ML model learns any residual from inclination/eccentricity).
        offset = np.array(
            [
                semi_major_km * np.cos(angle),
                semi_major_km * np.sin(angle),
                0.0,
            ]
        )
        return parent_xyz + offset

    # Standard bodies: use astropy ephemeris directly.
    for ephemeris in ("jpl", "builtin"):
        try:
            with solar_system_ephemeris.set(ephemeris):
                obj = get_body(body, time)
                cart = obj.cartesian
                return np.array(
                    [
                        cart.x.to(u.km).value,
                        cart.y.to(u.km).value,
                        cart.z.to(u.km).value,
                    ]
                )
        except (KeyError, ValueError):
            continue
    raise ValueError(f"Body '{body}' not found in any available ephemeris.")


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def compute_singular_distance(
    origin: str,
    target: str,
    date: Any,
) -> float:
    """Centre-to-centre distance between two bodies in km.

    Parameters
    ----------
    origin : str
        Origin body name (e.g. ``"earth"``).
    target : str
        Target body name (e.g. ``"moon"``).
    date : datetime-like
        Observation time.

    Returns
    -------
    float
        Scalar distance in km.
    """
    _require_astropy()
    origin = _resolve_body(origin)
    target = _resolve_body(target)

    time = Time(date)

    origin_xyz = _get_body_xyz(origin, time)
    target_xyz = _get_body_xyz(target, time)

    return float(np.linalg.norm(target_xyz - origin_xyz))


def _body_surface_xyz(
    body: str,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    time: "Time",
) -> np.ndarray:
    """Convert lat/lon on a body's surface to GCRS Cartesian xyz in km.

    Uses a spherical model with the body's mean volumetric radius,
    then rotates from body-fixed to inertial (GCRS) coordinates
    using the body's sidereal rotation period.  The rotation angle
    is measured from J2000.0.

    For Earth, this is equivalent to applying the Greenwich Mean
    Sidereal Time rotation.  For other bodies, it uses their
    sidereal rotation period from NASA fact sheets.

    Parameters
    ----------
    body : str
        Normalised body name (must be in ``_BODY_RADIUS_KM``).
    latitudes : np.ndarray
        Observer latitudes in degrees, shape ``(N,)``.
    longitudes : np.ndarray
        Observer longitudes in degrees, shape ``(N,)``.
    time : astropy.time.Time
        Observation time (used to compute rotation angle).

    Returns
    -------
    np.ndarray
        Shape ``(3, N)`` array of [x, y, z] positions in km (GCRS).
    """
    r = _BODY_RADIUS_KM[body]
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    cos_lat = np.cos(lat_rad)

    # Body-fixed Cartesian (before rotation).
    x_bf = r * cos_lat * np.cos(lon_rad)
    y_bf = r * cos_lat * np.sin(lon_rad)
    z_bf = r * np.sin(lat_rad)

    # Rotation angle: how far the body has rotated since J2000.0.
    period = _ROTATION_PERIOD_DAYS[body]
    elapsed_days = (time - Time("J2000.0")).jd
    theta = 2.0 * np.pi * (elapsed_days / period)

    # Rotate around the z-axis (equatorial plane rotation).
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_gcrs = cos_t * x_bf - sin_t * y_bf
    y_gcrs = sin_t * x_bf + cos_t * y_bf

    return np.array([x_gcrs, y_gcrs, z_bf])  # (3, N)


def compute_distance(
    origin: str,
    target: str,
    date: Any,
    latitudes: Any,
    longitudes: Any,
) -> np.ndarray:
    """Per-observer distance from *origin* surface to *target* centre.

    Observers are placed on the surface of the *origin* body at the
    given latitudes and longitudes using a spherical model with the
    body's mean volumetric radius.

    Parameters
    ----------
    origin : str
        Origin body name.
    target : str
        Target body name.
    date : datetime-like
        Observation time.
    latitudes : array-like
        Observer latitudes in degrees.
    longitudes : array-like
        Observer longitudes in degrees.

    Returns
    -------
    np.ndarray
        Distances in km, shape ``(N,)``.
    """
    _require_astropy()

    origin = _resolve_body(origin)
    target = _resolve_body(target)

    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)

    time = Time(date)
    target_xyz = _get_body_xyz(target, time)

    # Observer positions on the origin body surface, rotated to GCRS.
    obs_xyz = _body_surface_xyz(origin, latitudes, longitudes, time)  # (3, N)

    diff = obs_xyz - target_xyz[:, np.newaxis]  # (3, N)
    distances = np.linalg.norm(diff, axis=0)  # (N,)

    return distances


def compute_delta_distance(
    origin: str,
    target: str,
    date: Any,
    latitudes: Any,
    longitudes: Any,
) -> Any:
    """Per-observer distance minus the instantaneous maximum.

    Returns a non-positive array encoding the spatial variation in
    distance without the large constant offset.

    Parameters
    ----------
    origin, target : str
        Body names.
    date : datetime-like
        Observation time.
    latitudes, longitudes : array-like
        Observer coordinates in degrees.

    Returns
    -------
    np.ndarray
        Delta distances in km, shape ``(N,)``.
    """
    distances = compute_distance(origin, target, date, latitudes, longitudes)
    return distances - np.min(distances)


# ---------------------------------------------------------------------------
# Dynamic forcing maker
# ---------------------------------------------------------------------------


class DistanceForcingMaker:
    """Dynamic distance forcing maker with ``__getattr__`` dispatch.

    Resolves attribute names of the form::

        distance_from_{origin}_to_{target}
        singular_distance_from_{origin}_to_{target}
        delta_distance_from_{origin}_to_{target}

    and returns a callable ``(date, latitudes, longitudes) -> result``.

    The set of supported forcing names is not fixed -- any valid body
    pair is accepted.  The ``SUPPORTED`` property dynamically lists
    all combinations for discoverability.

    Examples
    --------
    >>> maker = DistanceForcingMaker()
    >>> d = maker.distance_from_earth_to_moon(date, lats, lons)
    >>> s = maker.singular_distance_from_mars_to_phobos(date, lats, lons)
    >>> dd = maker.delta_distance_from_earth_to_moon(date, lats, lons)
    """

    @property
    def SUPPORTED(self) -> set[str]:
        """Dynamically generate the set of supported attribute names.

        Lists all ``distance_from_{A}_to_{B}`` combinations (plus
        singular/delta variants) for known bodies.  This is for
        discoverability -- the actual dispatch accepts any valid pair.
        """
        names = set()
        for origin in sorted(KNOWN_BODIES):
            for target in sorted(KNOWN_BODIES):
                if origin == target:
                    continue
                names.add(f"distance_from_{origin}_to_{target}")
                names.add(f"singular_distance_from_{origin}_to_{target}")
                names.add(f"delta_distance_from_{origin}_to_{target}")
        return names

    def _parse_attr(self, name: str) -> tuple[str, str, str] | None:
        """Parse an attribute name into (variant, origin, target).

        Parameters
        ----------
        name : str
            Attribute name to parse.

        Returns
        -------
        tuple or None
            ``(variant, origin, target)`` where variant is one of
            ``"distance"``, ``"singular"``, ``"delta"``, or ``None``
            if the name does not match any pattern.
        """
        # Full-form patterns: distance_from_{origin}_to_{target}
        m = _RE_SINGULAR.match(name)
        if m:
            return ("singular", m.group(1), m.group(2))

        m = _RE_DELTA.match(name)
        if m:
            return ("delta", m.group(1), m.group(2))

        m = _RE_DISTANCE.match(name)
        if m:
            return ("distance", m.group(1), m.group(2))

        return None

    def __getattr__(self, name: str) -> Any:
        """Dynamically resolve distance forcing methods.

        Parses the attribute name to extract the variant (distance /
        singular / delta) and the body pair, then returns a bound
        callable.

        Parameters
        ----------
        name : str
            Attribute name, e.g. ``"distance_from_earth_to_moon"``.

        Returns
        -------
        callable
            ``(date, latitudes, longitudes) -> result``

        Raises
        ------
        AttributeError
            If the name does not match any recognised pattern or
            contains unknown body names.
        """
        parsed = self._parse_attr(name)
        if parsed is None:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'.  "
                f"Expected pattern: [singular_|delta_]distance_from_{{origin}}_to_{{target}}"
            )

        variant, origin, target = parsed

        # Validate body names early so errors are clear.
        _resolve_body(origin)
        _resolve_body(target)

        if origin == target:
            raise AttributeError(f"Origin and target must differ, got '{origin}' for both.")

        # Build and return the appropriate callable.
        # Singular takes only date; distance and delta take date + grid.
        _bound: Any
        if variant == "singular":

            def _singular(date: Any, *_args: Any) -> Any:
                return compute_singular_distance(origin, target, date)

            _bound = _singular
        elif variant == "delta":

            def _delta(date: Any, latitudes: Any, longitudes: Any) -> Any:
                return compute_delta_distance(origin, target, date, latitudes, longitudes)

            _bound = _delta
        else:

            def _dist(date: Any, latitudes: Any, longitudes: Any) -> Any:
                return compute_distance(origin, target, date, latitudes, longitudes)

            _bound = _dist

        # Cache on the instance to avoid repeated __getattr__ overhead.
        _bound.__name__ = name
        _bound.__doc__ = f"{variant.title()} distance from {origin} to {target} in km."
        object.__setattr__(self, name, _bound)
        return _bound


# ---------------------------------------------------------------------------
# Anemoi-datasets Source
# ---------------------------------------------------------------------------


class DistanceForcingsSource(Source):
    """Anemoi-datasets source producing distance forcings on a grid.

    Follows the same ``template`` + ``param`` pattern as the built-in
    ``forcings`` source.  The *template* is another source in the
    recipe whose grid (lat/lon) is reused, so any grid geometry is
    supported without hard-coding ``nlat`` / ``nlon``.

    Accepts arbitrary body pairs via the ``param`` list, using the
    same naming convention as :class:`DistanceForcingMaker`::

        [singular_|delta_]distance_from_{origin}_to_{target}

    Distance forcings are always time-varying (they depend on the
    relative positions of the bodies at each date).

    Example YAML
    -------------
    ::

        input:
          join:
            - mars: &mars
                arcomars:
                  dataset: ananyo01/ARCO-MACDA
                  param: [sp, skt]
            - distance_forcings:
                template: *mars
                param:
                  - distance_from_mars_to_phobos
                  - delta_distance_from_mars_to_phobos
                  - singular_distance_from_mars_to_phobos

    Or with the ``${...}`` reference syntax::

        input:
          join:
            - arcomars:
                dataset: ananyo01/ARCO-MACDA
                param: [sp, skt]
            - distance_forcings:
                template: ${input.join.0.arcomars}
                param:
                  - distance_from_mars_to_phobos

    Parameters
    ----------
    context : Any
        Pipeline context (from anemoi-datasets).
    template : Any
        Reference to another source in the recipe (resolved by
        anemoi-datasets to a source/dataset object whose grid
        geometry is inherited).
    param : list[str]
        Which distance forcing parameters to compute.
    """

    emoji = "\U0001f30d"  # globe

    def __init__(
        self,
        context: Any,
        template: Any,
        param: list[str] | str,
    ) -> None:
        super().__init__(context)
        self.template = template
        self.param = param if isinstance(param, list) else [param]

        # Validate all param names parse correctly.
        self._maker = DistanceForcingMaker()
        for p in self.param:
            parsed = self._maker._parse_attr(p)
            if parsed is None:
                raise ValueError(
                    f"Unknown distance_forcings param '{p}'.  "
                    f"Expected pattern: "
                    f"[singular_|delta_]distance_from_{{origin}}_to_{{target}}"
                )
            _, origin, target = parsed
            _resolve_body(origin)
            _resolve_body(target)

    def execute_valid_dates(self, dates: DateList) -> Any:
        """Compute distance forcing fields for the requested dates."""
        from anemoi.transform.fields import new_field_from_numpy
        from anemoi.transform.fields import new_fieldlist_from_list
        from earthkit.data import from_source

        self.context.trace(self.emoji, f"distance_forcings({self.param})")

        # Use earthkit forcings source to get a template field that
        # carries the grid geometry from the template source/dataset.
        # cos_local_time is time-varying, which is what we want.
        template_fields = from_source(
            "forcings",
            source_or_dataset=self.template,
            date=[dates[0]],
            param=["cos_local_time"],
        )
        template_field = template_fields[0]

        # Extract lat/lon from the template grid.
        flat_lats = template_field.grid_points()[0]
        flat_lons = template_field.grid_points()[1]
        n_points = len(flat_lats)

        result = []
        for date in dates:
            if not isinstance(date, dt.datetime):
                date = dt.datetime.fromisoformat(str(date))

            for param in self.param:
                parsed = self._maker._parse_attr(param)
                variant, origin, target = parsed

                if variant == "singular":
                    value = compute_singular_distance(origin, target, date)
                    values = np.full(n_points, value)
                elif variant == "delta":
                    values = compute_delta_distance(
                        origin,
                        target,
                        date,
                        flat_lats,
                        flat_lons,
                    )
                else:
                    values = compute_distance(
                        origin,
                        target,
                        date,
                        flat_lats,
                        flat_lons,
                    )

                result.append(
                    new_field_from_numpy(
                        values,
                        template=template_field,
                        param=param,
                        variable=param,
                        valid_datetime=date.isoformat(),
                        units="km",
                    )
                )

        return new_fieldlist_from_list(result)
