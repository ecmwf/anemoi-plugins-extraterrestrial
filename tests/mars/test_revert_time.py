# SPDX-FileCopyrightText: 2026 Anemoi contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Mars time-reversion tool.

Tests the pure conversion functions and the in-memory Dataset
reversion.  No network access or HuggingFace credentials required.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from anemoi.plugins.extraterrestrial.mars.revert_time import _GRID_EPOCH
from anemoi.plugins.extraterrestrial.mars.revert_time import MARS_YEAR_SOLS
from anemoi.plugins.extraterrestrial.mars.revert_time import datetime64_to_sol
from anemoi.plugins.extraterrestrial.mars.revert_time import revert_nc_time
from anemoi.plugins.extraterrestrial.mars.revert_time import revert_time_axis
from anemoi.plugins.extraterrestrial.mars.revert_time import sol_to_mars_year
from anemoi.plugins.extraterrestrial.mars.revert_time import sol_to_solar_longitude


# ---------------------------------------------------------------------------
# datetime64_to_sol  (24 Earth-hours = 1 sol)
# ---------------------------------------------------------------------------
class TestDatetime64ToSol:
    """Test the datetime-to-sol conversion."""

    def test_epoch_is_sol_zero(self):
        """Grid epoch should map to sol 0."""
        sols = datetime64_to_sol(np.array([_GRID_EPOCH]))
        assert np.isclose(sols[0], 0.0)

    def test_one_day_is_one_sol(self):
        """24 hours after epoch = sol 1."""
        t = _GRID_EPOCH + np.timedelta64(24, "h")
        sols = datetime64_to_sol(np.array([t]))
        assert np.isclose(sols[0], 1.0)

    def test_half_day_is_half_sol(self):
        """12 hours after epoch = sol 0.5."""
        t = _GRID_EPOCH + np.timedelta64(12, "h")
        sols = datetime64_to_sol(np.array([t]))
        assert np.isclose(sols[0], 0.5)

    def test_two_hours_is_one_twelfth_sol(self):
        """2 hours = 1/12 day = 1/12 sol."""
        t = _GRID_EPOCH + np.timedelta64(2, "h")
        sols = datetime64_to_sol(np.array([t]))
        assert np.isclose(sols[0], 1.0 / 12.0)

    def test_six_hours_is_quarter_sol(self):
        """6 hours = 0.25 day = 0.25 sol (works for 6h model output)."""
        t = _GRID_EPOCH + np.timedelta64(6, "h")
        sols = datetime64_to_sol(np.array([t]))
        assert np.isclose(sols[0], 0.25)

    def test_multiple_steps(self):
        """Array of times should convert element-wise."""
        times = _GRID_EPOCH + np.arange(5, dtype="int64") * np.timedelta64(6, "h")
        sols = datetime64_to_sol(times)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(sols, expected)

    def test_large_offset(self):
        """1000 days from epoch = sol 1000."""
        t = _GRID_EPOCH + np.timedelta64(1000, "D")
        sols = datetime64_to_sol(np.array([t]))
        assert np.isclose(sols[0], 1000.0)

    def test_roundtrip_2h_grid(self):
        """Forward (sols_to_regular_grid, 2h) then back should give identity."""
        from anemoi.plugins.extraterrestrial.mars.source import sols_to_regular_grid

        n = 100
        grid = sols_to_regular_grid(n, frequency_h=2)
        sols = datetime64_to_sol(grid)
        expected = np.arange(n) / 12.0  # 12 steps per day
        np.testing.assert_allclose(sols, expected, atol=1e-10)

    def test_roundtrip_1h_grid(self):
        """Forward (sols_to_regular_grid, 1h) then back should give identity."""
        from anemoi.plugins.extraterrestrial.mars.source import sols_to_regular_grid

        n = 48
        grid = sols_to_regular_grid(n, frequency_h=1)
        sols = datetime64_to_sol(grid)
        expected = np.arange(n) / 24.0  # 24 steps per day
        np.testing.assert_allclose(sols, expected, atol=1e-10)

    def test_works_any_resolution(self):
        """Should work for arbitrary time spacing (e.g. 6h model output)."""
        # Simulate a 6h output: 4 steps per day
        times = _GRID_EPOCH + np.arange(8, dtype="int64") * np.timedelta64(6, "h")
        sols = datetime64_to_sol(times)
        expected = np.arange(8) / 4.0  # 0, 0.25, 0.5, ..., 1.75
        np.testing.assert_allclose(sols, expected)


# ---------------------------------------------------------------------------
# sol_to_solar_longitude
# ---------------------------------------------------------------------------
class TestSolToSolarLongitude:
    """Test solar longitude computation from sol values."""

    def test_sol_zero_is_ls_zero(self):
        ls = sol_to_solar_longitude(np.array([0.0]))
        assert np.isclose(ls[0], 0.0)

    def test_half_year_is_ls_180(self):
        ls = sol_to_solar_longitude(np.array([MARS_YEAR_SOLS / 2.0]))
        assert np.isclose(ls[0], 180.0, atol=1e-4)

    def test_quarter_year_is_ls_90(self):
        ls = sol_to_solar_longitude(np.array([MARS_YEAR_SOLS / 4.0]))
        assert np.isclose(ls[0], 90.0, atol=1e-4)

    def test_full_year_wraps_to_zero(self):
        ls = sol_to_solar_longitude(np.array([MARS_YEAR_SOLS]))
        assert np.isclose(ls[0], 0.0, atol=1e-6)

    def test_range_always_valid(self):
        sols = np.linspace(0, MARS_YEAR_SOLS * 5, 500)
        ls = sol_to_solar_longitude(sols)
        assert np.all(ls >= 0.0)
        assert np.all(ls < 360.0)

    def test_vectorised(self):
        sols = np.array([0.0, MARS_YEAR_SOLS / 4, MARS_YEAR_SOLS / 2])
        ls = sol_to_solar_longitude(sols)
        np.testing.assert_allclose(ls, [0.0, 90.0, 180.0], atol=1e-4)


# ---------------------------------------------------------------------------
# sol_to_mars_year
# ---------------------------------------------------------------------------
class TestSolToMarsYear:
    """Test Mars Year number computation."""

    def test_sol_zero_is_base_my(self):
        my = sol_to_mars_year(np.array([0.0]), base_my=24)
        assert my[0] == 24

    def test_one_year_later(self):
        my = sol_to_mars_year(np.array([MARS_YEAR_SOLS]), base_my=24)
        assert my[0] == 25

    def test_mid_year_stays_same(self):
        my = sol_to_mars_year(np.array([MARS_YEAR_SOLS / 2]), base_my=24)
        assert my[0] == 24

    def test_custom_base_my(self):
        my = sol_to_mars_year(np.array([0.0]), base_my=28)
        assert my[0] == 28

    def test_returns_int32(self):
        my = sol_to_mars_year(np.array([0.0]))
        assert my.dtype == np.int32

    def test_vectorised(self):
        sols = np.array([0.0, MARS_YEAR_SOLS, MARS_YEAR_SOLS * 3])
        my = sol_to_mars_year(sols, base_my=24)
        np.testing.assert_array_equal(my, [24, 25, 27])


# ---------------------------------------------------------------------------
# revert_time_axis (in-memory)
# ---------------------------------------------------------------------------
class TestRevertTimeAxis:
    """Test the in-memory time axis reversion."""

    def _make_dataset(self, n_steps: int, frequency_h: int = 2) -> xr.Dataset:
        """Create a synthetic dataset with the Earth-time grid."""
        from anemoi.plugins.extraterrestrial.mars.source import sols_to_regular_grid

        times = sols_to_regular_grid(n_steps, frequency_h=frequency_h)
        data = np.random.rand(n_steps, 4, 8).astype(np.float32)
        return xr.Dataset(
            {"temperature": (["time", "lat", "lon"], data)},
            coords={"time": times, "lat": np.arange(4), "lon": np.arange(8)},
        )

    def test_time_renamed_to_sol(self):
        ds = self._make_dataset(24)
        result = revert_time_axis(ds)
        assert "sol" in result.dims
        assert "time" not in result.dims

    def test_sol_values_correct_2h(self):
        """12 steps of 2h each = 1 sol."""
        ds = self._make_dataset(24, frequency_h=2)
        result = revert_time_axis(ds)
        expected_sols = np.arange(24) / 12.0
        np.testing.assert_allclose(result.sol.values, expected_sols, atol=1e-10)

    def test_sol_values_correct_1h(self):
        """24 steps of 1h each = 1 sol."""
        ds = self._make_dataset(48, frequency_h=1)
        result = revert_time_axis(ds)
        expected_sols = np.arange(48) / 24.0
        np.testing.assert_allclose(result.sol.values, expected_sols, atol=1e-10)

    def test_sol_starts_at_zero(self):
        ds = self._make_dataset(12)
        result = revert_time_axis(ds)
        assert np.isclose(result.sol.values[0], 0.0)

    def test_data_preserved(self):
        ds = self._make_dataset(12)
        original_data = ds["temperature"].values.copy()
        result = revert_time_axis(ds)
        np.testing.assert_array_equal(result["temperature"].values, original_data)

    def test_other_coords_preserved(self):
        ds = self._make_dataset(12)
        result = revert_time_axis(ds)
        assert "lat" in result.coords
        assert "lon" in result.coords

    def test_sol_has_attrs(self):
        ds = self._make_dataset(12)
        result = revert_time_axis(ds)
        assert result.sol.attrs["units"] == "sols"

    def test_add_ls(self):
        ds = self._make_dataset(24)
        result = revert_time_axis(ds, add_ls=True)
        assert "Ls" in result.coords
        assert len(result.Ls) == 24
        assert np.all(result.Ls.values >= 0.0)
        assert np.all(result.Ls.values < 360.0)

    def test_add_ls_attrs(self):
        ds = self._make_dataset(12)
        result = revert_time_axis(ds, add_ls=True)
        assert result.Ls.attrs["units"] == "degrees"

    def test_add_mars_year(self):
        ds = self._make_dataset(24)
        result = revert_time_axis(ds, add_mars_year=True)
        assert "mars_year" in result.coords
        assert result.mars_year.values[0] == 24

    def test_no_ls_by_default(self):
        ds = self._make_dataset(12)
        result = revert_time_axis(ds)
        assert "Ls" not in result.coords

    def test_no_mars_year_by_default(self):
        ds = self._make_dataset(12)
        result = revert_time_axis(ds)
        assert "mars_year" not in result.coords

    def test_raises_without_time_dim(self):
        ds = xr.Dataset(
            {"temperature": (["lat", "lon"], np.ones((4, 8)))},
            coords={"lat": np.arange(4), "lon": np.arange(8)},
        )
        with pytest.raises(ValueError, match="no 'time' dimension"):
            revert_time_axis(ds)

    def test_all_options_together(self):
        ds = self._make_dataset(24)
        result = revert_time_axis(ds, add_ls=True, add_mars_year=True)
        assert "sol" in result.dims
        assert "Ls" in result.coords
        assert "mars_year" in result.coords
        assert "temperature" in result.data_vars

    def test_works_with_6h_output(self):
        """Should handle 6-hourly model output (4 steps per sol)."""
        # Simulate 6h output directly (not from sols_to_regular_grid)
        times = _GRID_EPOCH + np.arange(8, dtype="int64") * np.timedelta64(6, "h")
        ds = xr.Dataset(
            {"temperature": (["time", "lat", "lon"], np.ones((8, 4, 8)))},
            coords={"time": times, "lat": np.arange(4), "lon": np.arange(8)},
        )
        result = revert_time_axis(ds)
        expected_sols = np.arange(8) / 4.0
        np.testing.assert_allclose(result.sol.values, expected_sols)


# ---------------------------------------------------------------------------
# revert_nc_time (file-level)
# ---------------------------------------------------------------------------
class TestRevertNcTime:
    """Test the file-level NetCDF reversion."""

    def _make_nc(self, tmp_path, n_steps: int = 24) -> str:
        """Create a temporary NetCDF file with synthetic Earth-time grid."""
        from anemoi.plugins.extraterrestrial.mars.source import sols_to_regular_grid

        times = sols_to_regular_grid(n_steps, frequency_h=2)
        ds = xr.Dataset(
            {
                "temperature": (
                    ["time", "lat", "lon"],
                    np.random.rand(n_steps, 4, 8).astype(np.float32),
                )
            },
            coords={"time": times, "lat": np.arange(4), "lon": np.arange(8)},
        )
        path = str(tmp_path / "output.nc")
        ds.to_netcdf(path)
        return path

    def test_creates_output_file(self, tmp_path):
        input_path = self._make_nc(tmp_path)
        output_path = str(tmp_path / "output_mars.nc")
        revert_nc_time(input_path, output_path=output_path)
        assert Path(output_path).exists()

    def test_default_output_path(self, tmp_path):
        input_path = self._make_nc(tmp_path)
        revert_nc_time(input_path)
        assert (tmp_path / "output_mars.nc").exists()

    def test_output_has_sol_dim(self, tmp_path):
        input_path = self._make_nc(tmp_path)
        output_path = str(tmp_path / "reverted.nc")
        revert_nc_time(input_path, output_path=output_path)
        ds = xr.open_dataset(output_path)
        assert "sol" in ds.dims
        assert "time" not in ds.dims

    def test_output_has_correct_sol_values(self, tmp_path):
        input_path = self._make_nc(tmp_path, n_steps=12)
        output_path = str(tmp_path / "reverted.nc")
        revert_nc_time(input_path, output_path=output_path)
        ds = xr.open_dataset(output_path)
        expected = np.arange(12) / 12.0
        np.testing.assert_allclose(ds.sol.values, expected, atol=1e-10)

    def test_with_ls_and_mars_year(self, tmp_path):
        input_path = self._make_nc(tmp_path)
        output_path = str(tmp_path / "reverted.nc")
        revert_nc_time(input_path, output_path=output_path, add_ls=True, add_mars_year=True)
        ds = xr.open_dataset(output_path)
        assert "Ls" in ds.coords
        assert "mars_year" in ds.coords

    def test_raises_file_exists(self, tmp_path):
        input_path = self._make_nc(tmp_path)
        output_path = str(tmp_path / "existing.nc")
        Path(output_path).touch()
        with pytest.raises(FileExistsError):
            revert_nc_time(input_path, output_path=output_path)

    def test_overwrite_flag(self, tmp_path):
        input_path = self._make_nc(tmp_path)
        output_path = str(tmp_path / "existing.nc")
        Path(output_path).touch()
        revert_nc_time(input_path, output_path=output_path, overwrite=True)
        ds = xr.open_dataset(output_path)
        assert "sol" in ds.dims

    def test_data_preserved_through_file(self, tmp_path):
        from anemoi.plugins.extraterrestrial.mars.source import sols_to_regular_grid

        n = 12
        times = sols_to_regular_grid(n, frequency_h=2)
        original_data = np.arange(n * 4 * 8, dtype=np.float32).reshape(n, 4, 8)
        ds = xr.Dataset(
            {"temperature": (["time", "lat", "lon"], original_data)},
            coords={"time": times, "lat": np.arange(4), "lon": np.arange(8)},
        )
        input_path = str(tmp_path / "input.nc")
        ds.to_netcdf(input_path)

        output_path = str(tmp_path / "reverted.nc")
        revert_nc_time(input_path, output_path=output_path)
        result = xr.open_dataset(output_path)
        np.testing.assert_array_equal(result["temperature"].values, original_data)
