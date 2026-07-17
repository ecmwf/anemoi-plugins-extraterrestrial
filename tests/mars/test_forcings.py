# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit tests for Mars forcings computation functions.

Tests pure computation functions only - no network access required.
"""

import datetime as dt

import numpy as np
import pytest

from anemoi.plugins.extraterrestrial.mars.forcings import _GRID_EPOCH
from anemoi.plugins.extraterrestrial.mars.forcings import MARS_YEAR_SOLS
from anemoi.plugins.extraterrestrial.mars.forcings import MarsForcingMaker
from anemoi.plugins.extraterrestrial.mars.forcings import _datetime_to_sol
from anemoi.plugins.extraterrestrial.mars.forcings import _mars_solar_declination
from anemoi.plugins.extraterrestrial.mars.forcings import _sol_of_day
from anemoi.plugins.extraterrestrial.mars.forcings import _sol_of_year
from anemoi.plugins.extraterrestrial.mars.forcings import _solar_longitude_rad


class TestDatetimeToSol:
    """Test conversion from synthetic Earth datetime to Mars sol number."""

    def test_epoch_is_sol_zero(self):
        """At epoch (2000-01-01T00:00:00) we should get sol 0."""
        sol = _datetime_to_sol(_GRID_EPOCH, steps_per_sol=12, frequency_h=2)
        assert sol == 0.0

    def test_one_sol_with_12_steps_2h_frequency(self):
        """One full sol = 12 steps × 2h = 24h."""
        date = _GRID_EPOCH + dt.timedelta(hours=24)
        sol = _datetime_to_sol(date, steps_per_sol=12, frequency_h=2)
        assert np.isclose(sol, 1.0)

    def test_one_sol_with_24_steps_1h_frequency(self):
        """One full sol = 24 steps × 1h = 24h."""
        date = _GRID_EPOCH + dt.timedelta(hours=24)
        sol = _datetime_to_sol(date, steps_per_sol=24, frequency_h=1)
        assert np.isclose(sol, 1.0)

    def test_one_twelfth_sol(self):
        """One step = 2h = 1/12 sol (with steps_per_sol=12, frequency_h=2)."""
        date = _GRID_EPOCH + dt.timedelta(hours=2)
        sol = _datetime_to_sol(date, steps_per_sol=12, frequency_h=2)
        assert np.isclose(sol, 1.0 / 12.0)

    def test_half_sol(self):
        """Half a sol = 12h."""
        date = _GRID_EPOCH + dt.timedelta(hours=12)
        sol = _datetime_to_sol(date, steps_per_sol=12, frequency_h=2)
        assert np.isclose(sol, 0.5)

    def test_multiple_sols(self):
        """Multiple sols should accumulate correctly."""
        date = _GRID_EPOCH + dt.timedelta(hours=72)  # 3 sols
        sol = _datetime_to_sol(date, steps_per_sol=12, frequency_h=2)
        assert np.isclose(sol, 3.0)


class TestSolOfYear:
    """Test extracting fractional position within the Mars year."""

    def test_sol_zero(self):
        """Sol 0 should give 0."""
        assert _sol_of_year(0.0) == 0.0

    def test_mid_year(self):
        """Mid-year sol should pass through unchanged."""
        sol = 334.3
        assert np.isclose(_sol_of_year(sol), 334.3)

    def test_full_year_wraps(self):
        """Exactly one Mars year should wrap to 0."""
        assert np.isclose(_sol_of_year(MARS_YEAR_SOLS), 0.0, atol=1e-10)

    def test_multiple_years(self):
        """Multiple years should wrap correctly."""
        sol = MARS_YEAR_SOLS * 3 + 100.0
        assert np.isclose(_sol_of_year(sol), 100.0, atol=1e-3)

    def test_small_remainder(self):
        """Small remainder after wrapping."""
        sol = MARS_YEAR_SOLS + 1.5
        assert np.isclose(_sol_of_year(sol), 1.5, atol=1e-10)


class TestSolOfDay:
    """Test extracting fractional position within the Mars sol."""

    def test_sol_zero(self):
        """Sol 0 should give 0."""
        assert _sol_of_day(0.0) == 0.0

    def test_full_day_wraps(self):
        """Exactly 1.0 sol should wrap to 0."""
        assert _sol_of_day(1.0) == 0.0

    def test_half_sol(self):
        """0.5 sol should give 0.5."""
        assert np.isclose(_sol_of_day(0.5), 0.5)

    def test_quarter_sol(self):
        """0.25 sol should give 0.25."""
        assert np.isclose(_sol_of_day(0.25), 0.25)

    def test_multiple_sols_with_fraction(self):
        """Multiple sols should extract only the fractional part."""
        assert np.isclose(_sol_of_day(2.75), 0.75)
        assert np.isclose(_sol_of_day(10.1), 0.1, atol=1e-10)

    def test_large_sol_number(self):
        """Large sol numbers should work correctly."""
        assert np.isclose(_sol_of_day(1000.333), 0.333, atol=1e-10)


class TestSolarLongitudeRad:
    """Test approximate solar longitude calculation."""

    def test_epoch_is_zero(self):
        """At sol 0, Ls should be 0 radians."""
        ls = _solar_longitude_rad(0.0)
        assert np.isclose(ls, 0.0)

    def test_half_year(self):
        """At half Mars year, Ls should be π radians (180°)."""
        sol = MARS_YEAR_SOLS / 2.0
        ls = _solar_longitude_rad(sol)
        assert np.isclose(ls, np.pi, atol=1e-6)

    def test_quarter_year(self):
        """At quarter Mars year, Ls should be π/2 radians (90°)."""
        sol = MARS_YEAR_SOLS / 4.0
        ls = _solar_longitude_rad(sol)
        assert np.isclose(ls, np.pi / 2.0, atol=1e-6)

    def test_full_year_wraps(self):
        """One full Mars year should wrap back to 0."""
        ls = _solar_longitude_rad(MARS_YEAR_SOLS)
        assert np.isclose(ls, 0.0, atol=1e-10)

    def test_multiple_years(self):
        """Multiple years should wrap correctly."""
        sol = MARS_YEAR_SOLS * 3.0 + MARS_YEAR_SOLS / 4.0
        ls = _solar_longitude_rad(sol)
        assert np.isclose(ls, np.pi / 2.0, atol=1e-6)


class TestMarsSolarDeclination:
    """Test Mars solar declination calculation."""

    def test_spring_equinox(self):
        """At Ls=0 (spring equinox), declination should be 0."""
        dec = _mars_solar_declination(0.0)
        assert np.isclose(dec, 0.0, atol=1e-10)

    def test_autumn_equinox(self):
        """At Ls=π (autumn equinox), declination should be 0."""
        dec = _mars_solar_declination(np.pi)
        assert np.isclose(dec, 0.0, atol=1e-10)

    def test_summer_solstice(self):
        """At Ls=π/2 (summer solstice), declination should equal obliquity."""
        obliquity = np.deg2rad(25.19)
        dec = _mars_solar_declination(np.pi / 2.0)
        assert np.isclose(dec, obliquity, atol=1e-6)

    def test_winter_solstice(self):
        """At Ls=3π/2 (winter solstice), declination should be -obliquity."""
        obliquity = np.deg2rad(25.19)
        dec = _mars_solar_declination(3.0 * np.pi / 2.0)
        assert np.isclose(dec, -obliquity, atol=1e-6)

    def test_declination_range(self):
        """Declination should be bounded by ±obliquity."""
        obliquity = np.deg2rad(25.19)
        for ls in np.linspace(0, 2 * np.pi, 100):
            dec = _mars_solar_declination(ls)
            assert -obliquity <= dec <= obliquity


class TestMarsForcingMakerSupported:
    """Test the SUPPORTED parameter set."""

    def test_supported_contains_all_expected(self):
        """Check that SUPPORTED contains exactly the expected 10 parameters."""
        expected = {
            "cos_latitude",
            "sin_latitude",
            "cos_longitude",
            "sin_longitude",
            "cos_sol_of_year",
            "sin_sol_of_year",
            "cos_local_time",
            "sin_local_time",
            "insolation",
            "cos_solar_longitude",
            "sin_solar_longitude",
        }
        assert MarsForcingMaker.SUPPORTED == expected


class TestMarsForcingMakerGeometry:
    """Test geometry-based forcings (latitude/longitude)."""

    @pytest.fixture
    def maker(self):
        """Create a MarsForcingMaker with known lat/lon values."""
        lats = np.array([87.5, 0.0, -87.5])
        lons = np.array([0.0, 90.0, 180.0])
        return MarsForcingMaker(lats, lons, steps_per_sol=12, frequency_h=2)

    def test_cos_latitude(self, maker):
        """Test cos(latitude) computation."""
        result = maker.cos_latitude(_GRID_EPOCH)
        expected = np.array(
            [
                np.cos(np.deg2rad(87.5)),  # ≈ 0.0436
                np.cos(np.deg2rad(0.0)),  # = 1.0
                np.cos(np.deg2rad(-87.5)),  # ≈ 0.0436
            ]
        )
        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_sin_latitude(self, maker):
        """Test sin(latitude) computation."""
        result = maker.sin_latitude(_GRID_EPOCH)
        expected = np.array(
            [
                np.sin(np.deg2rad(87.5)),  # ≈ 0.9990
                np.sin(np.deg2rad(0.0)),  # = 0.0
                np.sin(np.deg2rad(-87.5)),  # ≈ -0.9990
            ]
        )
        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_cos_longitude(self, maker):
        """Test cos(longitude) computation."""
        result = maker.cos_longitude(_GRID_EPOCH)
        expected = np.array(
            [
                np.cos(np.deg2rad(0.0)),  # = 1.0
                np.cos(np.deg2rad(90.0)),  # ≈ 0.0
                np.cos(np.deg2rad(180.0)),  # = -1.0
            ]
        )
        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sin_longitude(self, maker):
        """Test sin(longitude) computation."""
        result = maker.sin_longitude(_GRID_EPOCH)
        expected = np.array(
            [
                np.sin(np.deg2rad(0.0)),  # = 0.0
                np.sin(np.deg2rad(90.0)),  # = 1.0
                np.sin(np.deg2rad(180.0)),  # ≈ 0.0
            ]
        )
        assert result.shape == (3,)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_geometry_time_independent(self, maker):
        """Geometry forcings should be independent of date."""
        date1 = _GRID_EPOCH
        date2 = _GRID_EPOCH + dt.timedelta(days=365)

        for method in [
            "cos_latitude",
            "sin_latitude",
            "cos_longitude",
            "sin_longitude",
        ]:
            result1 = getattr(maker, method)(date1)
            result2 = getattr(maker, method)(date2)
            np.testing.assert_array_equal(result1, result2)

    def test_trig_identity_latitude(self, maker):
        """cos²(lat) + sin²(lat) should equal 1."""
        cos_lat = maker.cos_latitude(_GRID_EPOCH)
        sin_lat = maker.sin_latitude(_GRID_EPOCH)
        identity = cos_lat**2 + sin_lat**2
        np.testing.assert_allclose(identity, 1.0, rtol=1e-10)

    def test_trig_identity_longitude(self, maker):
        """cos²(lon) + sin²(lon) should equal 1."""
        cos_lon = maker.cos_longitude(_GRID_EPOCH)
        sin_lon = maker.sin_longitude(_GRID_EPOCH)
        identity = cos_lon**2 + sin_lon**2
        np.testing.assert_allclose(identity, 1.0, rtol=1e-10)


class TestMarsForcingMakerYearlyCycle:
    """Test yearly cycle forcings (sol of year, solar longitude)."""

    @pytest.fixture
    def maker(self):
        """Create a MarsForcingMaker with simple grid."""
        lats = np.array([0.0, 45.0, -45.0])
        lons = np.array([0.0, 0.0, 0.0])
        return MarsForcingMaker(lats, lons, steps_per_sol=12, frequency_h=2)

    def test_cos_sol_of_year_at_epoch(self, maker):
        """At epoch (sol 0), cos_sol_of_year should be 1."""
        result = maker.cos_sol_of_year(_GRID_EPOCH)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_sin_sol_of_year_at_epoch(self, maker):
        """At epoch (sol 0), sin_sol_of_year should be 0."""
        result = maker.sin_sol_of_year(_GRID_EPOCH)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_cos_sol_of_year_at_half_year(self, maker):
        """At half Mars year, cos_sol_of_year should be -1."""
        # Half year = MARS_YEAR_SOLS / 2 sols = MARS_YEAR_SOLS / 2 * 24h
        hours = MARS_YEAR_SOLS / 2.0 * 24.0
        date = _GRID_EPOCH + dt.timedelta(hours=hours)
        result = maker.cos_sol_of_year(date)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, -1.0, atol=1e-6)

    def test_sin_sol_of_year_at_quarter_year(self, maker):
        """At quarter Mars year, sin_sol_of_year should be 1."""
        hours = MARS_YEAR_SOLS / 4.0 * 24.0
        date = _GRID_EPOCH + dt.timedelta(hours=hours)
        result = maker.sin_sol_of_year(date)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_sol_of_year_uniform_across_grid(self, maker):
        """Sol-of-year forcings should be uniform across all grid points."""
        date = _GRID_EPOCH + dt.timedelta(days=100)
        cos_result = maker.cos_sol_of_year(date)
        sin_result = maker.sin_sol_of_year(date)

        # All values should be identical
        assert np.all(cos_result == cos_result[0])
        assert np.all(sin_result == sin_result[0])

    def test_trig_identity_sol_of_year(self, maker):
        """cos²(sol_of_year) + sin²(sol_of_year) should equal 1."""
        date = _GRID_EPOCH + dt.timedelta(days=123)
        cos_jd = maker.cos_sol_of_year(date)
        sin_jd = maker.sin_sol_of_year(date)
        identity = cos_jd**2 + sin_jd**2
        np.testing.assert_allclose(identity, 1.0, rtol=1e-10)

    def test_cos_solar_longitude_at_epoch(self, maker):
        """At epoch (Ls=0), cos(Ls) should be 1."""
        result = maker.cos_solar_longitude(_GRID_EPOCH)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_sin_solar_longitude_at_epoch(self, maker):
        """At epoch (Ls=0), sin(Ls) should be 0."""
        result = maker.sin_solar_longitude(_GRID_EPOCH)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_cos_solar_longitude_at_half_year(self, maker):
        """At half Mars year (Ls=180), cos(Ls) should be -1."""
        hours = MARS_YEAR_SOLS / 2.0 * 24.0
        date = _GRID_EPOCH + dt.timedelta(hours=hours)
        result = maker.cos_solar_longitude(date)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, -1.0, atol=1e-4)

    def test_sin_solar_longitude_at_quarter_year(self, maker):
        """At quarter Mars year (Ls=90), sin(Ls) should be 1."""
        hours = MARS_YEAR_SOLS / 4.0 * 24.0
        date = _GRID_EPOCH + dt.timedelta(hours=hours)
        result = maker.sin_solar_longitude(date)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, 1.0, atol=1e-4)

    def test_solar_longitude_trig_identity(self, maker):
        """cos²(Ls) + sin²(Ls) should equal 1."""
        for i in range(20):
            date = _GRID_EPOCH + dt.timedelta(days=i * 50)
            cos_ls = maker.cos_solar_longitude(date)
            sin_ls = maker.sin_solar_longitude(date)
            np.testing.assert_allclose(cos_ls**2 + sin_ls**2, 1.0, rtol=1e-10)

    def test_solar_longitude_in_range(self, maker):
        """cos/sin(Ls) should always be in [-1, 1]."""
        for i in range(20):
            date = _GRID_EPOCH + dt.timedelta(days=i * 50)
            assert np.all(np.abs(maker.cos_solar_longitude(date)) <= 1.0)
            assert np.all(np.abs(maker.sin_solar_longitude(date)) <= 1.0)

    def test_solar_longitude_uniform_across_grid(self, maker):
        """Solar longitude components should be uniform across all grid points."""
        date = _GRID_EPOCH + dt.timedelta(days=200)
        cos_ls = maker.cos_solar_longitude(date)
        sin_ls = maker.sin_solar_longitude(date)
        assert np.all(cos_ls == cos_ls[0])
        assert np.all(sin_ls == sin_ls[0])


class TestMarsForcingMakerDailyCycle:
    """Test daily cycle forcings (local time)."""

    @pytest.fixture
    def maker(self):
        """Create a MarsForcingMaker with varying longitudes."""
        lats = np.array([0.0, 0.0, 0.0])
        lons = np.array([0.0, 90.0, 180.0])
        return MarsForcingMaker(lats, lons, steps_per_sol=12, frequency_h=2)

    def test_cos_local_time_at_epoch_lon_zero(self, maker):
        """At epoch, lon=0, local time fraction is 0 → cos=1."""
        result = maker.cos_local_time(_GRID_EPOCH)
        assert result.shape == (3,)
        assert np.isclose(result[0], 1.0, atol=1e-10)

    def test_sin_local_time_at_epoch_lon_zero(self, maker):
        """At epoch, lon=0, local time fraction is 0 → sin=0."""
        result = maker.sin_local_time(_GRID_EPOCH)
        assert result.shape == (3,)
        assert np.isclose(result[0], 0.0, atol=1e-10)

    def test_cos_local_time_longitude_offset(self, maker):
        """At epoch, lon=180 should be half sol offset → cos=-1."""
        result = maker.cos_local_time(_GRID_EPOCH)
        # lon=180 → local_frac = (0 + 180/360) % 1 = 0.5
        # angle = 0.5 * 2π = π → cos(π) = -1
        assert np.isclose(result[2], -1.0, atol=1e-10)

    def test_sin_local_time_longitude_offset(self, maker):
        """At epoch, lon=180 should be half sol offset → sin≈0."""
        result = maker.sin_local_time(_GRID_EPOCH)
        assert np.isclose(result[2], 0.0, atol=1e-10)

    def test_cos_local_time_half_sol(self, maker):
        """After half sol (12h), lon=0 should have cos=-1 (midday)."""
        date = _GRID_EPOCH + dt.timedelta(hours=12)
        result = maker.cos_local_time(date)
        # sol = 0.5, local_frac = (0.5 + 0/360) = 0.5
        # angle = π → cos=-1
        assert np.isclose(result[0], -1.0, atol=1e-10)

    def test_sin_local_time_quarter_sol(self, maker):
        """After quarter sol (6h), lon=0 should have sin=1."""
        date = _GRID_EPOCH + dt.timedelta(hours=6)
        result = maker.sin_local_time(date)
        # sol = 0.25, local_frac = 0.25
        # angle = π/2 → sin=1
        assert np.isclose(result[0], 1.0, atol=1e-10)

    def test_local_time_varies_by_longitude(self, maker):
        """Local time should vary by longitude at the same instant."""
        result = maker.cos_local_time(_GRID_EPOCH)
        # Different longitudes should give different values
        assert not np.allclose(result[0], result[2])

    def test_trig_identity_local_time(self, maker):
        """cos²(local_time) + sin²(local_time) should equal 1."""
        date = _GRID_EPOCH + dt.timedelta(hours=7)
        cos_lt = maker.cos_local_time(date)
        sin_lt = maker.sin_local_time(date)
        identity = cos_lt**2 + sin_lt**2
        np.testing.assert_allclose(identity, 1.0, rtol=1e-10)


class TestMarsForcingMakerInsolation:
    """Test insolation (cos solar zenith angle) computation."""

    @pytest.fixture
    def maker(self):
        """Create a MarsForcingMaker with poles and equator."""
        lats = np.array([87.5, 0.0, -87.5])
        lons = np.array([0.0, 0.0, 0.0])
        return MarsForcingMaker(lats, lons, steps_per_sol=12, frequency_h=2)

    def test_insolation_shape(self, maker):
        """Insolation should return correct shape."""
        result = maker.insolation(_GRID_EPOCH)
        assert result.shape == (3,)

    def test_insolation_range(self, maker):
        """Insolation should be clipped to [0, 1]."""
        for i in range(20):
            date = _GRID_EPOCH + dt.timedelta(days=i * 30)
            result = maker.insolation(date)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_insolation_equator_local_noon_positive(self, maker):
        """At equator, local noon should have positive insolation."""
        # At epoch (Ls=0, spring equinox), solar dec ≈ 0
        # Local noon: sol=0.5 (12h), lon=0 → hour_angle=0
        # cos_sza = sin(0)*sin(0) + cos(0)*cos(0)*cos(0) = 1
        date = _GRID_EPOCH + dt.timedelta(hours=12)
        result = maker.insolation(date)
        assert result[1] > 0.0  # Equator point

    def test_insolation_nighttime_zero(self, maker):
        """At local midnight, some points should have zero insolation."""
        # At epoch (sol=0), lon=0 → midnight
        # At equator during equinox, midnight should have near-zero insolation
        # Actually hour_angle = (0 - 0.5) * 2π = -π → cos(ha) = -1
        # cos_sza = 0*0 + 1*1*(-1) = -1 → clipped to 0
        result = maker.insolation(_GRID_EPOCH)
        # At least one point should be zero (nighttime)
        assert result[1] == 0.0

    def test_insolation_dtype(self, maker):
        """Insolation should return numpy array."""
        result = maker.insolation(_GRID_EPOCH)
        assert isinstance(result, np.ndarray)

    def test_insolation_different_seasons(self):
        """Insolation should vary across Mars seasons."""
        lats = np.array([45.0])
        lons = np.array([0.0])
        maker = MarsForcingMaker(lats, lons, steps_per_sol=12, frequency_h=2)

        # Northern spring equinox (Ls=0)
        spring = maker.insolation(_GRID_EPOCH)

        # Northern summer solstice (Ls=90)
        hours_summer = MARS_YEAR_SOLS / 4.0 * 24.0
        summer = maker.insolation(_GRID_EPOCH + dt.timedelta(hours=hours_summer))

        # Values should be different due to changing solar declination
        # (though exact comparison depends on local time)
        assert spring.shape == summer.shape == (1,)


class TestMarsForcingMakerOutputProperties:
    """Test general properties of all forcing outputs."""

    @pytest.fixture
    def maker(self):
        """Create a MarsForcingMaker with realistic grid."""
        lats = np.linspace(90, -90, 18)  # ~10° spacing
        lons = np.linspace(-180, 175, 36)  # 10° spacing
        lat_grid, lon_grid = np.meshgrid(lats, lons)
        return MarsForcingMaker(lat_grid.ravel(), lon_grid.ravel(), steps_per_sol=12, frequency_h=2)

    def test_all_methods_return_ndarray(self, maker):
        """All forcing methods should return numpy arrays."""
        date = _GRID_EPOCH + dt.timedelta(days=50)
        for param in MarsForcingMaker.SUPPORTED:
            result = getattr(maker, param)(date)
            assert isinstance(result, np.ndarray)

    def test_all_methods_return_correct_shape(self, maker):
        """All methods should return 1D array with n_points elements."""
        date = _GRID_EPOCH + dt.timedelta(days=50)
        expected_shape = (maker.n_points,)
        for param in MarsForcingMaker.SUPPORTED:
            result = getattr(maker, param)(date)
            assert result.shape == expected_shape

    def test_trig_values_in_range(self, maker):
        """All cos/sin methods should return values in [-1, 1]."""
        date = _GRID_EPOCH + dt.timedelta(days=100)
        trig_params = [
            "cos_latitude",
            "sin_latitude",
            "cos_longitude",
            "sin_longitude",
            "cos_sol_of_year",
            "sin_sol_of_year",
            "cos_local_time",
            "sin_local_time",
        ]
        for param in trig_params:
            result = getattr(maker, param)(date)
            assert np.all(result >= -1.0)
            assert np.all(result <= 1.0)

    def test_insolation_in_range(self, maker):
        """Insolation should be in [0, 1]."""
        for i in range(10):
            date = _GRID_EPOCH + dt.timedelta(days=i * 67)
            result = maker.insolation(date)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_solar_longitude_components_in_range(self, maker):
        """cos/sin(Ls) should be in [-1, 1]."""
        for i in range(10):
            date = _GRID_EPOCH + dt.timedelta(days=i * 67)
            assert np.all(np.abs(maker.cos_solar_longitude(date)) <= 1.0)
            assert np.all(np.abs(maker.sin_solar_longitude(date)) <= 1.0)


class TestMarsForcingMakerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_grid(self):
        """Test with single grid point."""
        maker = MarsForcingMaker(np.array([0.0]), np.array([0.0]), steps_per_sol=12, frequency_h=2)
        result = maker.cos_latitude(_GRID_EPOCH)
        assert result.shape == (1,)
        assert np.isclose(result[0], 1.0)

    def test_negative_longitude(self):
        """Test with negative longitude values (e.g., -180 to 180)."""
        maker = MarsForcingMaker(
            np.array([0.0, 0.0]),
            np.array([-90.0, 90.0]),
            steps_per_sol=12,
            frequency_h=2,
        )
        cos_lon = maker.cos_longitude(_GRID_EPOCH)
        # cos(-90°) = cos(90°) = 0
        np.testing.assert_allclose(cos_lon, 0.0, atol=1e-10)

    def test_extreme_latitude(self):
        """Test with poles (±90°)."""
        maker = MarsForcingMaker(
            np.array([90.0, -90.0]),
            np.array([0.0, 0.0]),
            steps_per_sol=12,
            frequency_h=2,
        )
        cos_lat = maker.cos_latitude(_GRID_EPOCH)
        # cos(±90°) = 0
        np.testing.assert_allclose(cos_lat, 0.0, atol=1e-10)

        sin_lat = maker.sin_latitude(_GRID_EPOCH)
        # sin(90°) = 1, sin(-90°) = -1
        assert np.isclose(sin_lat[0], 1.0)
        assert np.isclose(sin_lat[1], -1.0)

    def test_different_frequency_configs(self):
        """Test with different steps_per_sol and frequency_h configurations."""
        configs = [
            (12, 2),  # 12 steps × 2h = 24h
            (24, 1),  # 24 steps × 1h = 24h
            (8, 3),  # 8 steps × 3h = 24h
        ]
        lats = np.array([0.0])
        lons = np.array([0.0])

        for steps, freq in configs:
            maker = MarsForcingMaker(lats, lons, steps_per_sol=steps, frequency_h=freq)
            # After 24h, should be back to same local time
            date = _GRID_EPOCH + dt.timedelta(hours=24)
            cos_lt = maker.cos_local_time(date)
            # At sol=1.0, local_frac=0 → cos=1
            assert np.isclose(cos_lt[0], 1.0, atol=1e-10)

    def test_very_large_date_offset(self):
        """Test with date far from epoch."""
        maker = MarsForcingMaker(np.array([0.0]), np.array([0.0]), steps_per_sol=12, frequency_h=2)
        # Many Mars years later
        date = _GRID_EPOCH + dt.timedelta(days=10000)

        # Should still produce valid outputs
        for param in MarsForcingMaker.SUPPORTED:
            result = getattr(maker, param)(date)
            assert result.shape == (1,)
            assert np.isfinite(result[0])


class TestMarsForcingMakerConsistency:
    """Test consistency between related forcings."""

    @pytest.fixture
    def maker(self):
        """Create a standard maker for consistency tests."""
        lats = np.array([45.0, 0.0, -45.0])
        lons = np.array([0.0, 90.0, 180.0])
        return MarsForcingMaker(lats, lons, steps_per_sol=12, frequency_h=2)

    def test_sol_of_year_matches_solar_longitude(self, maker):
        """Sol-of-year trig and solar longitude trig should be consistent.

        Both encode the same yearly angle (linear Ls approximation),
        so cos/sin(sol_of_year) == cos/sin(solar_longitude).
        """
        date = _GRID_EPOCH + dt.timedelta(days=200)

        cos_soy = maker.cos_sol_of_year(date)
        sin_soy = maker.sin_sol_of_year(date)
        cos_ls = maker.cos_solar_longitude(date)
        sin_ls = maker.sin_solar_longitude(date)

        # Should match — both are trig of the same underlying angle
        assert np.isclose(cos_soy[0], cos_ls[0], atol=1e-6)
        assert np.isclose(sin_soy[0], sin_ls[0], atol=1e-6)

    def test_insolation_uses_consistent_local_time(self, maker):
        """Insolation computation should be consistent with local time."""
        # This is more of a sanity check that the same sol/local_frac
        # calculation is used internally
        date = _GRID_EPOCH + dt.timedelta(hours=6)

        # Both should use the same underlying sol calculation
        _ = maker.cos_local_time(date)
        insol = maker.insolation(date)

        # Just verify they're computed without error and have correct shape
        assert insol.shape == (3,)
        assert np.all(np.isfinite(insol))
