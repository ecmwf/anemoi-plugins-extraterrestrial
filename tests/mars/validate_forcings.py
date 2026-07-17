#!/usr/bin/env python3
# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Quick validation of all Mars forcing variables.

Creates a small dataset (4 grid points, 10 dates spanning ~half a Mars year)
and checks that every variable has physically correct values.
No network access required — uses MarsForcingMaker directly.
"""

import datetime as dt
import sys

import numpy as np

from anemoi.plugins.extraterrestrial.mars.forcings import _GRID_EPOCH
from anemoi.plugins.extraterrestrial.mars.forcings import MARS_YEAR_SOLS
from anemoi.plugins.extraterrestrial.mars.forcings import MarsForcingMaker
from anemoi.plugins.extraterrestrial.mars.forcings import _datetime_to_sol

# ── tiny grid ──────────────────────────────────────────────────────────────
LATS = np.array([45.0, 0.0, -45.0, 87.5])
LONS = np.array([0.0, 90.0, -90.0, 180.0])
STEPS_PER_SOL = 12
FREQUENCY_H = 2

# 10 dates spread over ~half a Mars year
N_DATES = 10
HALF_YEAR_H = MARS_YEAR_SOLS / 2.0 * 24.0  # hours
DATES = [_GRID_EPOCH + dt.timedelta(hours=HALF_YEAR_H * i / (N_DATES - 1)) for i in range(N_DATES)]

maker = MarsForcingMaker(LATS, LONS, steps_per_sol=STEPS_PER_SOL, frequency_h=FREQUENCY_H)

errors = []


def check(condition, msg):
    if not condition:
        errors.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")


print("=" * 72)
print(f"Validating {len(MarsForcingMaker.SUPPORTED)} forcings")
print(f"  Grid:  {len(LATS)} points")
print(f"  Dates: {N_DATES} ({DATES[0]} … {DATES[-1]})")
print(f"  Sol range: 0 … {_datetime_to_sol(DATES[-1], STEPS_PER_SOL, FREQUENCY_H):.1f}")
print("=" * 72)

# ── 1. Spatial forcings: time-independent ──────────────────────────────────
print("\n── Spatial (time-invariant) forcings ──")
for name in ("cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude"):
    vals = [getattr(maker, name)(d) for d in DATES]
    for i in range(1, len(vals)):
        check(
            np.array_equal(vals[0], vals[i]),
            f"{name} identical at date 0 vs date {i}",
        )
    check(np.all(np.abs(vals[0]) <= 1.0), f"{name} in [-1, 1]")

# trig identities
cos_lat = maker.cos_latitude(DATES[0])
sin_lat = maker.sin_latitude(DATES[0])
check(
    np.allclose(cos_lat**2 + sin_lat**2, 1.0),
    "cos²(lat) + sin²(lat) == 1",
)
cos_lon = maker.cos_longitude(DATES[0])
sin_lon = maker.sin_longitude(DATES[0])
check(
    np.allclose(cos_lon**2 + sin_lon**2, 1.0),
    "cos²(lon) + sin²(lon) == 1",
)

# ── 2. Sol-of-year: must VARY over half a Mars year ───────────────────────
print("\n── Sol-of-year (yearly cycle) forcings ──")
cos_vals = np.array([maker.cos_sol_of_year(d) for d in DATES])  # (N_DATES, n_pts)
sin_vals = np.array([maker.sin_sol_of_year(d) for d in DATES])

# Should be uniform across grid at each time step
for i, d in enumerate(DATES):
    check(
        np.all(cos_vals[i] == cos_vals[i, 0]),
        f"cos_sol_of_year uniform across grid at date {i}",
    )
    check(
        np.all(sin_vals[i] == sin_vals[i, 0]),
        f"sin_sol_of_year uniform across grid at date {i}",
    )

# Must VARY across time (this is the bug we're checking for)
cos_range = cos_vals[:, 0].max() - cos_vals[:, 0].min()
sin_range = sin_vals[:, 0].max() - sin_vals[:, 0].min()
check(
    cos_range > 1.5,
    f"cos_sol_of_year range={cos_range:.3f} > 1.5 (varies over half year)",
)
check(
    sin_range > 0.5,
    f"sin_sol_of_year range={sin_range:.3f} > 0.5 (varies over half year)",
)

# Boundary values: epoch → cos=1, sin=0; half year → cos≈-1
check(np.allclose(cos_vals[0, 0], 1.0, atol=1e-10), "cos_sol_of_year(epoch) == 1")
check(np.allclose(sin_vals[0, 0], 0.0, atol=1e-10), "sin_sol_of_year(epoch) == 0")
check(np.allclose(cos_vals[-1, 0], -1.0, atol=1e-4), "cos_sol_of_year(half year) ≈ -1")

# Trig identity
check(
    np.allclose(cos_vals**2 + sin_vals**2, 1.0),
    "cos²(sol_of_year) + sin²(sol_of_year) == 1 at all dates",
)

# ── 3. Solar longitude (cos/sin components) ───────────────────────────────
print("\n── Solar longitude (cos/sin) ──")
cos_ls = np.array([maker.cos_solar_longitude(d) for d in DATES])
sin_ls = np.array([maker.sin_solar_longitude(d) for d in DATES])

# Boundary values: epoch → Ls=0 → cos=1, sin=0; half year → Ls=180 → cos=-1
check(np.allclose(cos_ls[0, 0], 1.0, atol=1e-10), "cos(Ls(epoch)) == 1")
check(np.allclose(sin_ls[0, 0], 0.0, atol=1e-10), "sin(Ls(epoch)) == 0")
check(np.allclose(cos_ls[-1, 0], -1.0, atol=1e-4), "cos(Ls(half year)) ≈ -1")

# Trig identity
check(
    np.allclose(cos_ls**2 + sin_ls**2, 1.0),
    "cos²(Ls) + sin²(Ls) == 1 at all dates",
)

# Must vary over time
cos_ls_range = cos_ls[:, 0].max() - cos_ls[:, 0].min()
check(
    cos_ls_range > 1.5,
    f"cos(Ls) range={cos_ls_range:.3f} > 1.5 (varies over half year)",
)

# Uniform across grid
for i in range(N_DATES):
    check(np.all(cos_ls[i] == cos_ls[i, 0]), f"cos(Ls) uniform across grid at date {i}")
    check(np.all(sin_ls[i] == sin_ls[i, 0]), f"sin(Ls) uniform across grid at date {i}")

# ── 4. Local time: varies across grid AND time ────────────────────────────
print("\n── Local time (daily cycle) forcings ──")
cos_lt = np.array([maker.cos_local_time(d) for d in DATES])
sin_lt = np.array([maker.sin_local_time(d) for d in DATES])

# Must vary across grid points (different longitudes → different local time)
for i in range(N_DATES):
    check(
        not np.all(cos_lt[i] == cos_lt[i, 0]),
        f"cos_local_time varies across grid at date {i}",
    )

# Trig identity
check(
    np.allclose(cos_lt**2 + sin_lt**2, 1.0),
    "cos²(local_time) + sin²(local_time) == 1 at all dates",
)

# Range over all dates × grid points
check(cos_lt.min() < -0.5 and cos_lt.max() > 0.5, "cos_local_time spans wide range")
check(sin_lt.min() < -0.5 and sin_lt.max() > 0.5, "sin_local_time spans wide range")

# ── 5. Insolation ─────────────────────────────────────────────────────────
print("\n── Insolation ──")
insol = np.array([maker.insolation(d) for d in DATES])
check(np.all(insol >= 0.0), "insolation >= 0 everywhere")
check(np.all(insol <= 1.0), "insolation <= 1 everywhere")
check(insol.max() > 0.3, "insolation has daytime values > 0.3")
# Should have some nighttime zeros
check(np.any(insol == 0.0), "insolation has nighttime zeros")

# ── 6. Shape checks ──────────────────────────────────────────────────────
print("\n── Shape checks ──")
for param in sorted(MarsForcingMaker.SUPPORTED):
    v = getattr(maker, param)(DATES[0])
    check(v.shape == (len(LATS),), f"{param} shape == ({len(LATS)},)")

# ── 7. Constancy check (simulates what the dataset builder does) ──────────
print("\n── Constancy detection (simulated) ──")
first_values = {}
is_constant = {}
for param in sorted(MarsForcingMaker.SUPPORTED):
    vals = [getattr(maker, param)(d) for d in DATES]
    first_values[param] = vals[0]
    is_constant[param] = all(np.array_equal(vals[0], v) for v in vals[1:])

expected_constant = {"cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude"}
expected_varying = MarsForcingMaker.SUPPORTED - expected_constant

for p in sorted(expected_constant):
    check(is_constant[p], f"{p} detected as constant (correct)")

for p in sorted(expected_varying):
    check(not is_constant[p], f"{p} detected as time-varying (correct)")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
if errors:
    print(f"FAILED: {len(errors)} check(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
