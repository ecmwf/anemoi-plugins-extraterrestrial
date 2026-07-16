# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Unit tests for the Mars source plugin.

These tests do NOT require network access or HuggingFace credentials.
They use synthetic xarray Datasets to test the pure logic functions.
"""

import numpy as np
import pytest
import xarray as xr

from anemoi.plugins.extraterrestrial.mars.source import _DROP_VARS
from anemoi.plugins.extraterrestrial.mars.source import _KNOWN_STORES
from anemoi.plugins.extraterrestrial.mars.source import _RENAME_MACDA
from anemoi.plugins.extraterrestrial.mars.source import CANONICAL_EMARS
from anemoi.plugins.extraterrestrial.mars.source import CANONICAL_MACDA
from anemoi.plugins.extraterrestrial.mars.source import CANONICAL_OPENMARS
from anemoi.plugins.extraterrestrial.mars.source import _merge_openmars_eras
from anemoi.plugins.extraterrestrial.mars.source import _normalise_dataset
from anemoi.plugins.extraterrestrial.mars.source import _open_arco_zarr
from anemoi.plugins.extraterrestrial.mars.source import _process_emars
from anemoi.plugins.extraterrestrial.mars.source import _sort_and_fill_gaps
from anemoi.plugins.extraterrestrial.mars.source import sols_to_regular_grid


class TestSolsToRegularGrid:
    """Test the sols_to_regular_grid function."""

    def test_returns_datetime64_array(self):
        """Should return numpy.datetime64 array."""
        result = sols_to_regular_grid(10)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.dtype("datetime64[ns]")

    def test_correct_length(self):
        """Should return array of requested length."""
        for n in [1, 10, 100, 1000]:
            result = sols_to_regular_grid(n)
            assert len(result) == n

    def test_starts_at_epoch(self):
        """Should start at 2000-01-01T00:00:00."""
        result = sols_to_regular_grid(10)
        expected_epoch = np.datetime64("2000-01-01T00:00:00", "ns")
        assert result[0] == expected_epoch

    def test_frequency_2h_macda_openmars(self):
        """Steps should be exactly 2 hours apart for MACDA/OpenMars."""
        result = sols_to_regular_grid(10, frequency_h=2)
        expected_step = np.timedelta64(2, "h")

        # Check all consecutive differences
        diffs = np.diff(result)
        assert all(diffs == expected_step)

        # Check specific values
        epoch = np.datetime64("2000-01-01T00:00:00", "ns")
        assert result[1] == epoch + np.timedelta64(2, "h")
        assert result[2] == epoch + np.timedelta64(4, "h")
        assert result[9] == epoch + np.timedelta64(18, "h")

    def test_frequency_1h_emars(self):
        """Steps should be exactly 1 hour apart for EMARS."""
        result = sols_to_regular_grid(10, frequency_h=1)
        expected_step = np.timedelta64(1, "h")

        # Check all consecutive differences
        diffs = np.diff(result)
        assert all(diffs == expected_step)

        # Check specific values
        epoch = np.datetime64("2000-01-01T00:00:00", "ns")
        assert result[1] == epoch + np.timedelta64(1, "h")
        assert result[2] == epoch + np.timedelta64(2, "h")
        assert result[9] == epoch + np.timedelta64(9, "h")

    def test_large_n(self):
        """Should handle large arrays correctly."""
        n = 100_000
        result = sols_to_regular_grid(n, frequency_h=2)
        assert len(result) == n

        epoch = np.datetime64("2000-01-01T00:00:00", "ns")
        expected_last = epoch + np.timedelta64(2 * (n - 1), "h")
        assert result[-1] == expected_last


class TestKnownStoresCatalogue:
    """Test the _KNOWN_STORES catalogue structure."""

    def test_all_datasets_exist(self):
        """All 3 expected datasets should be in the catalogue."""
        expected_datasets = {CANONICAL_MACDA, CANONICAL_OPENMARS, CANONICAL_EMARS}
        assert set(_KNOWN_STORES.keys()) == expected_datasets

    def test_macda_metadata(self):
        """MACDA should have correct metadata."""
        info = _KNOWN_STORES[CANONICAL_MACDA]
        assert info["nlat"] == 36
        assert info["nlon"] == 72
        assert info["steps_per_sol"] == 12
        assert info["frequency_h"] == 2
        # Both backends configured
        assert info["hf"]["repo"] == "ananyo01/ARCO-MACDA"
        assert info["hf"]["default"] == "macda_combined.zarr"
        assert info["earthmover"]["repo"] == "arco-planetary/ARCO-MACDA"

    def test_openmars_metadata(self):
        """OpenMars should have correct metadata."""
        info = _KNOWN_STORES[CANONICAL_OPENMARS]
        assert info["nlat"] == 36
        assert info["nlon"] == 72
        assert info["steps_per_sol"] == 12
        assert info["frequency_h"] == 2
        assert info["hf"]["repo"] == "ananyo01/ARCO-OpenMars"
        assert info["hf"]["default"] == "openmars_unified.zarr"
        assert info["earthmover"]["repo"] == "arco-planetary/ARCO-OpenMARS"
        # Earthmover: eras are separate groups
        assert set(info["earthmover"]["era_groups"]) == {"my24", "my28"}

    def test_emars_metadata(self):
        """EMARS should have correct metadata."""
        info = _KNOWN_STORES[CANONICAL_EMARS]
        assert info["nlat"] == 36
        assert info["nlon"] == 60
        assert info["steps_per_sol"] == 24
        assert info["frequency_h"] == 1
        assert info["hf"]["repo"] == "ananyo01/ARCO-EMARS"
        assert info["earthmover"]["repo"] == "arco-planetary/ARCO-EMARS"
        # Earthmover: default to ensemble mean
        assert info["earthmover"]["default_group"] == "mean"

    def test_all_have_required_keys(self):
        """All datasets should have required top-level metadata keys."""
        required_keys = {
            "nlat",
            "nlon",
            "steps_per_sol",
            "frequency_h",
            "hf",
            "earthmover",
        }
        for dataset_id, info in _KNOWN_STORES.items():
            assert required_keys.issubset(info.keys()), f"Missing keys in {dataset_id}"


class TestNormaliseDataset:
    """Test the _normalise_dataset id-resolution helper."""

    def test_canonical_short_names_pass_through(self):
        assert _normalise_dataset("ARCO-MACDA") == CANONICAL_MACDA
        assert _normalise_dataset("ARCO-OpenMars") == CANONICAL_OPENMARS
        assert _normalise_dataset("ARCO-EMARS") == CANONICAL_EMARS

    def test_hf_qualified_ids_map_to_canonical(self):
        assert _normalise_dataset("ananyo01/ARCO-MACDA") == CANONICAL_MACDA
        assert _normalise_dataset("ananyo01/ARCO-OpenMars") == CANONICAL_OPENMARS
        assert _normalise_dataset("ananyo01/ARCO-EMARS") == CANONICAL_EMARS

    def test_earthmover_qualified_ids_map_to_canonical(self):
        assert _normalise_dataset("arco-planetary/ARCO-MACDA") == CANONICAL_MACDA
        assert _normalise_dataset("arco-planetary/ARCO-OpenMARS") == CANONICAL_OPENMARS
        assert _normalise_dataset("arco-planetary/ARCO-EMARS") == CANONICAL_EMARS

    def test_case_insensitive(self):
        assert _normalise_dataset("arco-macda") == CANONICAL_MACDA
        assert _normalise_dataset("ARCO-openmars") == CANONICAL_OPENMARS

    def test_openmars_capitalisation_variants(self):
        """Both 'OpenMars' (HF) and 'OpenMARS' (Earthmover) resolve."""
        assert _normalise_dataset("ARCO-OpenMars") == CANONICAL_OPENMARS
        assert _normalise_dataset("ARCO-OpenMARS") == CANONICAL_OPENMARS

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Unknown Mars dataset id"):
            _normalise_dataset("some/other-dataset")


class TestBackendDispatch:
    """Test that _open_arco_zarr dispatches to the right backend."""

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            _open_arco_zarr("ARCO-MACDA", backend="nope")

    def test_hf_backend_calls_hf_opener(self, monkeypatch):
        """backend='hf' (the default) should call _open_hf_zarr."""
        import anemoi.plugins.extraterrestrial.mars.source as src_mod

        called: dict = {}

        def fake_hf(canonical):
            called["backend"] = "hf"
            called["canonical"] = canonical
            return "hf-dataset"

        def fake_em(canonical):  # pragma: no cover - should not be called
            called["backend"] = "earthmover"
            return "em-dataset"

        monkeypatch.setattr(src_mod, "_open_hf_zarr", fake_hf)
        monkeypatch.setattr(src_mod, "_open_em_zarr", fake_em)

        result = src_mod._open_arco_zarr("ananyo01/ARCO-MACDA")
        assert result == "hf-dataset"
        assert called == {"backend": "hf", "canonical": CANONICAL_MACDA}

    def test_earthmover_backend_calls_em_opener(self, monkeypatch):
        """backend='earthmover' should call _open_em_zarr."""
        import anemoi.plugins.extraterrestrial.mars.source as src_mod

        called: dict = {}

        def fake_hf(canonical):  # pragma: no cover - should not be called
            called["backend"] = "hf"
            return "hf-dataset"

        def fake_em(canonical):
            called["backend"] = "earthmover"
            called["canonical"] = canonical
            return "em-dataset"

        monkeypatch.setattr(src_mod, "_open_hf_zarr", fake_hf)
        monkeypatch.setattr(src_mod, "_open_em_zarr", fake_em)

        # Short-name id + earthmover backend
        result = src_mod._open_arco_zarr("ARCO-EMARS", backend="earthmover")
        assert result == "em-dataset"
        assert called == {"backend": "earthmover", "canonical": CANONICAL_EMARS}

    def test_earthmover_via_qualified_id(self, monkeypatch):
        """Passing the earthmover-qualified id also routes correctly."""
        import anemoi.plugins.extraterrestrial.mars.source as src_mod

        called: dict = {}
        monkeypatch.setattr(
            src_mod,
            "_open_hf_zarr",
            lambda *a, **k: pytest.fail("HF should not be called"),
        )
        monkeypatch.setattr(
            src_mod,
            "_open_em_zarr",
            lambda canonical: called.setdefault("canonical", canonical) or "em",
        )

        src_mod._open_arco_zarr("arco-planetary/ARCO-OpenMARS", backend="earthmover")
        assert called["canonical"] == CANONICAL_OPENMARS


class TestRenameMacda:
    """Test the _RENAME_MACDA mapping."""

    def test_temperature_mapping(self):
        """Temperature should map to 't'."""
        assert _RENAME_MACDA["temp"] == "t"

    def test_wind_mappings(self):
        """Wind components should map correctly."""
        assert _RENAME_MACDA["uwind"] == "u"
        assert _RENAME_MACDA["vwind"] == "v"

    def test_pressure_mappings(self):
        """Pressure variables should map correctly."""
        assert _RENAME_MACDA["psurf"] == "sp"
        assert _RENAME_MACDA["omega"] == "w"

    def test_temperature_surface(self):
        """Surface temperature should map to 'skt'."""
        assert _RENAME_MACDA["tsurf"] == "skt"

    def test_geopotential(self):
        """Geopotential should map to 'z'."""
        assert _RENAME_MACDA["geop"] == "z"

    def test_radiation_mappings(self):
        """Radiation fluxes should map correctly."""
        assert _RENAME_MACDA["swflux"] == "ssrd"
        assert _RENAME_MACDA["lwflux"] == "strd"

    def test_mars_specific_variables(self):
        """Mars-specific variables should be preserved or renamed."""
        assert _RENAME_MACDA["co2ice"] == "co2ice"
        assert _RENAME_MACDA["coldust"] == "tcdo"
        assert _RENAME_MACDA["dustmmr"] == "dust"

    def test_all_values_are_valid_shortnames(self):
        """All target names should be valid ECMWF/CF shortnames."""
        # These are the expected standard names
        valid_targets = {
            "t",
            "u",
            "v",
            "sp",
            "skt",
            "z",
            "w",
            "ssrd",
            "strd",
            "co2ice",
            "tcdo",
            "dust",
        }
        assert set(_RENAME_MACDA.values()) == valid_targets

    def test_no_duplicate_values(self):
        """No two source variables should map to the same target."""
        values = list(_RENAME_MACDA.values())
        assert len(values) == len(set(values))


class TestMergeOpenMarsEras:
    """Test the _merge_openmars_eras function."""

    def test_merge_single_variable(self):
        """Should merge era-prefixed variables into single continuous one."""
        # Create synthetic data: MY24-27 has data for first half, MY28-35 for second
        time = np.arange(24)
        lat = np.arange(36)
        lon = np.arange(72)

        # MY24-27 has data for first 12 steps, NaN after
        my24_27_data = np.ones((24, 36, 72), dtype=np.float32)
        my24_27_data[12:, :, :] = np.nan

        # MY28-35 has NaN for first 12 steps, data after
        my28_35_data = np.ones((24, 36, 72), dtype=np.float32) * 2.0
        my28_35_data[:12, :, :] = np.nan

        ds = xr.Dataset(
            {
                "MY24-27_temp": (["time", "lat", "lon"], my24_27_data),
                "MY28-35_temp": (["time", "lat", "lon"], my28_35_data),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )

        result = _merge_openmars_eras(ds)

        # Should have single variable 't'
        assert "t" in result
        assert "MY24-27_temp" not in result
        assert "MY28-35_temp" not in result

        # First half should have value 1.0 (from MY24-27)
        assert np.allclose(result["t"].values[0, :, :], 1.0)
        assert np.allclose(result["t"].values[11, :, :], 1.0)

        # Second half should have value 2.0 (from MY28-35)
        assert np.allclose(result["t"].values[12, :, :], 2.0)
        assert np.allclose(result["t"].values[23, :, :], 2.0)

    def test_merge_multiple_variables(self):
        """Should merge all era-prefixed variables."""
        time = np.arange(12)
        lat = np.arange(36)
        lon = np.arange(72)

        # Create data for multiple variables
        temp_24_27 = np.ones((12, 36, 72), dtype=np.float32)
        temp_28_35 = np.ones((12, 36, 72), dtype=np.float32) * 2.0
        u_24_27 = np.ones((12, 36, 72), dtype=np.float32) * 3.0
        u_28_35 = np.ones((12, 36, 72), dtype=np.float32) * 4.0

        ds = xr.Dataset(
            {
                "MY24-27_temp": (["time", "lat", "lon"], temp_24_27),
                "MY28-35_temp": (["time", "lat", "lon"], temp_28_35),
                "MY24-27_u": (["time", "lat", "lon"], u_24_27),
                "MY28-35_u": (["time", "lat", "lon"], u_28_35),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )

        result = _merge_openmars_eras(ds)

        # Should have merged variables with standard names
        assert "t" in result
        assert "u" in result

        # Original era-prefixed variables should be removed
        assert "MY24-27_temp" not in result
        assert "MY28-35_temp" not in result
        assert "MY24-27_u" not in result
        assert "MY28-35_u" not in result

    def test_fillna_preserves_gap_regions(self):
        """NaN should be preserved where BOTH eras are NaN."""
        time = np.arange(12)
        lat = np.arange(36)
        lon = np.arange(72)

        # Both eras have NaN in the middle region (gap)
        my24_27_data = np.ones((12, 36, 72), dtype=np.float32)
        my24_27_data[5:7, :, :] = np.nan  # Gap

        my28_35_data = np.ones((12, 36, 72), dtype=np.float32) * 2.0
        my28_35_data[5:7, :, :] = np.nan  # Same gap

        ds = xr.Dataset(
            {
                "MY24-27_temp": (["time", "lat", "lon"], my24_27_data),
                "MY28-35_temp": (["time", "lat", "lon"], my28_35_data),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )

        result = _merge_openmars_eras(ds)

        # Gap region should still be NaN
        assert np.all(np.isnan(result["t"].values[5, :, :]))
        assert np.all(np.isnan(result["t"].values[6, :, :]))

        # Non-gap regions should have data (filled from first era)
        assert np.allclose(result["t"].values[0, :, :], 1.0)
        assert np.allclose(result["t"].values[11, :, :], 1.0)

    def test_all_expected_standard_names_produced(self):
        """All expected standard variable names should be produced."""
        time = np.arange(6)
        lat = np.arange(36)
        lon = np.arange(72)

        # Create all OpenMars variables
        variables = ["temp", "u", "v", "ps", "tsurf", "co2ice", "dustcol"]
        ds_vars = {}
        for var in variables:
            for era in ["MY24-27", "MY28-35"]:
                data = np.ones((6, 36, 72), dtype=np.float32)
                ds_vars[f"{era}_{var}"] = (["time", "lat", "lon"], data)

        ds = xr.Dataset(ds_vars, coords={"time": time, "lat": lat, "lon": lon})
        result = _merge_openmars_eras(ds)

        # Expected standard names
        expected = {"t", "u", "v", "sp", "skt", "co2ice", "tcdo"}
        assert set(result.data_vars.keys()) == expected

    def test_original_era_variables_removed(self):
        """Original era-prefixed variables should be removed."""
        time = np.arange(6)
        lat = np.arange(36)
        lon = np.arange(72)

        ds = xr.Dataset(
            {
                "MY24-27_temp": (["time", "lat", "lon"], np.ones((6, 36, 72))),
                "MY28-35_temp": (["time", "lat", "lon"], np.ones((6, 36, 72))),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )

        result = _merge_openmars_eras(ds)

        # No era-prefixed variables should remain
        for var in result.data_vars:
            assert not str(var).startswith("MY24-27_")
            assert not str(var).startswith("MY28-35_")


class TestProcessEmars:
    """Test the _process_emars function."""

    def test_strips_anal_mean_prefix(self):
        """Should strip 'anal_mean_' prefix from variable names."""
        time = np.arange(6)
        pfull = np.array([1.0, 2.0, 3.0])
        lat = np.arange(36)
        lon = np.arange(60)

        ds = xr.Dataset(
            {
                "anal_mean_T": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 3, 36, 60)),
                ),
                "anal_mean_U": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 3, 36, 60)),
                ),
            },
            coords={"time": time, "pfull": pfull, "lat": lat, "lon": lon},
        )

        result = _process_emars(ds)

        # Prefixes should be stripped
        assert "anal_mean_T" not in result
        assert "anal_mean_U" not in result

    def test_renames_variables(self):
        """Should rename EMARS variables to standard names."""
        time = np.arange(6)
        pfull = np.array([1.0, 2.0, 3.0])
        lat = np.arange(36)
        lon = np.arange(60)

        ds = xr.Dataset(
            {
                "anal_mean_T": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 3, 36, 60)),
                ),
                "anal_mean_U": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 3, 36, 60)),
                ),
                "anal_mean_V": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 3, 36, 60)),
                ),
                "anal_mean_ps": (["time", "lat", "lon"], np.ones((6, 36, 60))),
            },
            coords={"time": time, "pfull": pfull, "lat": lat, "lon": lon},
        )

        result = _process_emars(ds)

        # Should have standard names
        assert "t" in result
        assert "u" in result
        assert "v" in result
        assert "sp" in result

    def test_drops_auxiliary_variables(self):
        """Should drop auxiliary time/coordinate variables."""
        time = np.arange(6)
        pfull = np.array([1.0, 2.0, 3.0])
        lat = np.arange(36)
        lon = np.arange(60)

        ds = xr.Dataset(
            {
                "anal_mean_T": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 3, 36, 60)),
                ),
                "anal_mean_Ls": (["time"], np.ones(6)),
                "anal_mean_earth_year": (["time"], np.ones(6)),
                "anal_mean_ak": (["pfull"], np.ones(3)),
                "anal_mean_bk": (["pfull"], np.ones(3)),
            },
            coords={"time": time, "pfull": pfull, "lat": lat, "lon": lon},
        )

        result = _process_emars(ds)

        # Auxiliary variables should be dropped
        assert "anal_mean_Ls" not in result
        assert "anal_mean_earth_year" not in result
        assert "anal_mean_ak" not in result
        assert "anal_mean_bk" not in result
        assert "Ls" not in result
        assert "earth_year" not in result

    def test_renames_pfull_to_level(self):
        """Should rename 'pfull' dimension to 'level'."""
        time = np.arange(6)
        pfull = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lat = np.arange(36)
        lon = np.arange(60)

        ds = xr.Dataset(
            {
                "anal_mean_T": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 5, 36, 60)),
                ),
            },
            coords={"time": time, "pfull": pfull, "lat": lat, "lon": lon},
        )

        result = _process_emars(ds)

        # pfull should be renamed to level
        assert "pfull" not in result.dims
        assert "level" in result.dims

    def test_level_has_integer_values(self):
        """Level coordinate should have integer 1..N values."""
        time = np.arange(6)
        pfull = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lat = np.arange(36)
        lon = np.arange(60)

        ds = xr.Dataset(
            {
                "anal_mean_T": (
                    ["time", "pfull", "lat", "lon"],
                    np.ones((6, 5, 36, 60)),
                ),
            },
            coords={"time": time, "pfull": pfull, "lat": lat, "lon": lon},
        )

        result = _process_emars(ds)

        # Level should be 1-indexed integers
        expected_levels = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(result.coords["level"].values, expected_levels)

    def test_surface_geopotential_renamed(self):
        """Surface_geopotential should be renamed to z_sfc."""
        time = np.arange(6)
        lat = np.arange(36)
        lon = np.arange(60)

        ds = xr.Dataset(
            {
                "anal_mean_Surface_geopotential": (["lat", "lon"], np.ones((36, 60))),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )

        result = _process_emars(ds)

        # Should be renamed to z_sfc
        assert "z_sfc" in result
        assert "Surface_geopotential" not in result
        assert "anal_mean_Surface_geopotential" not in result


class TestFillSolGaps:
    """Test the _sort_and_fill_gaps function."""

    def test_no_gaps_returns_unchanged(self):
        """Dataset with no gaps should be returned unchanged."""
        # Create continuous sol sequence
        time = np.arange(12) / 12.0  # 0, 1/12, 2/12, ..., 11/12
        lat = np.arange(36)
        lon = np.arange(72)

        ds = xr.Dataset(
            {
                "t": (["time", "lat", "lon"], np.ones((12, 36, 72))),
            },
            coords={"time": time, "lat": lat, "lon": lon},
        )

        result = _sort_and_fill_gaps(ds, time, steps_per_sol=12)

        # Should have same number of steps
        assert len(result.time) == 12
        # Data should be unchanged
        assert np.allclose(result["t"].values, 1.0)

    def test_detects_and_fills_gap(self):
        """Should detect gaps and insert NaN-filled steps."""
        # Create sol sequence with gap: 0-11/12, then jump to 100+0/12
        first_era = np.arange(12) / 12.0  # sols 0 to 11/12
        second_era = 100.0 + np.arange(12) / 12.0  # sols 100 to 100+11/12
        raw_sols = np.concatenate([first_era, second_era])

        lat = np.arange(36)
        lon = np.arange(72)

        ds = xr.Dataset(
            {
                "t": (["time", "lat", "lon"], np.ones((24, 36, 72))),
            },
            coords={"time": raw_sols, "lat": lat, "lon": lon},
        )

        result = _sort_and_fill_gaps(ds, raw_sols, steps_per_sol=12)

        # Should have more than original 24 steps (gap filled)
        assert len(result.time) > 24

        # Gap is ~99 sols = ~99 * 12 = ~1188 steps
        expected_gap_steps = int(round((100.0 - 1.0) * 12))
        expected_total = 24 + expected_gap_steps
        assert len(result.time) == expected_total

    def test_nan_values_in_gap(self):
        """Gap steps should have NaN values for all data variables."""
        # Create small gap for easier testing
        first_era = np.array([0.0, 1 / 12, 2 / 12])
        second_era = np.array([2.0, 2.0 + 1 / 12, 2.0 + 2 / 12])  # 2 sol gap
        raw_sols = np.concatenate([first_era, second_era])

        lat = np.arange(4)
        lon = np.arange(4)

        # Use distinct values to track real vs gap data
        data = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)

        ds = xr.Dataset(
            {
                "t": (["time", "lat", "lon"], data),
            },
            coords={"time": raw_sols, "lat": lat, "lon": lon},
        )

        result = _sort_and_fill_gaps(ds, raw_sols, steps_per_sol=12)

        # First 3 steps should have original data
        assert np.allclose(result["t"].values[0, :, :], data[0, :, :])
        assert np.allclose(result["t"].values[1, :, :], data[1, :, :])
        assert np.allclose(result["t"].values[2, :, :], data[2, :, :])

        # Gap steps (between index 3 and last 3) should be NaN
        gap_start = 3
        gap_end = len(result.time) - 3
        for i in range(gap_start, gap_end):
            assert np.all(np.isnan(result["t"].values[i, :, :]))

        # Last 3 steps should have original data
        assert np.allclose(result["t"].values[-3, :, :], data[3, :, :])
        assert np.allclose(result["t"].values[-2, :, :], data[4, :, :])
        assert np.allclose(result["t"].values[-1, :, :], data[5, :, :])

    def test_preserves_real_data_order(self):
        """Real data before and after gap should be preserved in order."""
        first_era = np.array([0.0, 1 / 12])
        second_era = np.array([5.0, 5.0 + 1 / 12])  # 5 sol gap
        raw_sols = np.concatenate([first_era, second_era])

        lat = np.arange(2)
        lon = np.arange(2)

        # Create data with identifiable values
        data = np.array(
            [
                [[1.0, 1.0], [1.0, 1.0]],  # step 0
                [[2.0, 2.0], [2.0, 2.0]],  # step 1
                [[3.0, 3.0], [3.0, 3.0]],  # step 2 (after gap)
                [[4.0, 4.0], [4.0, 4.0]],  # step 3 (after gap)
            ]
        )

        ds = xr.Dataset(
            {
                "t": (["time", "lat", "lon"], data),
            },
            coords={"time": raw_sols, "lat": lat, "lon": lon},
        )

        result = _sort_and_fill_gaps(ds, raw_sols, steps_per_sol=12)

        # First 2 steps should be preserved
        assert np.allclose(result["t"].values[0, :, :], 1.0)
        assert np.allclose(result["t"].values[1, :, :], 2.0)

        # Last 2 steps should be preserved
        assert np.allclose(result["t"].values[-2, :, :], 3.0)
        assert np.allclose(result["t"].values[-1, :, :], 4.0)

    def test_multiple_gaps(self):
        """Should handle multiple gaps correctly."""
        # Create sequence with two gaps
        era1 = np.array([0.0, 1 / 12])
        era2 = np.array([5.0, 5.0 + 1 / 12])  # gap of ~5 sols
        era3 = np.array([10.0, 10.0 + 1 / 12])  # another gap of ~5 sols
        raw_sols = np.concatenate([era1, era2, era3])

        lat = np.arange(2)
        lon = np.arange(2)

        ds = xr.Dataset(
            {
                "t": (["time", "lat", "lon"], np.ones((6, 2, 2))),
            },
            coords={"time": raw_sols, "lat": lat, "lon": lon},
        )

        result = _sort_and_fill_gaps(ds, raw_sols, steps_per_sol=12)

        # Should have filled both gaps
        # Original 6 steps + 2 gaps of ~5 sols each (~60 steps each)
        assert len(result.time) > 6


class TestLevelRenaming:
    """Test level dimension renaming logic."""

    def test_lev_dimension_renamed_to_level(self):
        """'lev' dimension with sigma values should be renamed to 'level'."""
        time = np.arange(6)
        lev = np.array([0.9995, 0.9982, 0.9951, 0.9901])  # sigma values
        lat = np.arange(36)
        lon = np.arange(72)

        ds = xr.Dataset(
            {
                "t": (["time", "lev", "lat", "lon"], np.ones((6, 4, 36, 72))),
            },
            coords={"time": time, "lev": lev, "lat": lat, "lon": lon},
        )

        # Simulate the renaming logic from _open_hf_zarr
        if "lev" in ds.dims:
            n_levels = len(ds["lev"])
            ds = ds.rename({"lev": "level"})
            ds = ds.assign_coords(level=("level", np.arange(1, n_levels + 1)))

        assert "lev" not in ds.dims
        assert "level" in ds.dims

    def test_level_has_integer_1_to_n(self):
        """Level coordinate should have integer values 1..N."""
        time = np.arange(6)
        lev = np.array([0.9995, 0.9982, 0.9951, 0.9901, 0.9801])
        lat = np.arange(36)
        lon = np.arange(72)

        ds = xr.Dataset(
            {
                "t": (["time", "lev", "lat", "lon"], np.ones((6, 5, 36, 72))),
            },
            coords={"time": time, "lev": lev, "lat": lat, "lon": lon},
        )

        # Simulate the renaming logic
        if "lev" in ds.dims:
            n_levels = len(ds["lev"])
            ds = ds.rename({"lev": "level"})
            ds = ds.assign_coords(level=("level", np.arange(1, n_levels + 1)))

        expected = np.array([1, 2, 3, 4, 5])
        assert np.array_equal(ds.coords["level"].values, expected)


class TestDropVarsCompleteness:
    """Test that _DROP_VARS contains all required OpenMars auxiliary variables."""

    def test_all_openmars_aux_vars_in_drop_set(self):
        """All OpenMars auxiliary variables should be in _DROP_VARS."""
        expected_vars = {
            "Ls",
            "MY_Ls",
            "MY24-27_Ls",
            "MY28-35_Ls",
            "MY24-27_MY",
            "MY28-35_MY",
        }
        assert expected_vars.issubset(_DROP_VARS)

    def test_ls_variables(self):
        """Solar longitude variables should be in drop set."""
        assert "Ls" in _DROP_VARS
        assert "MY_Ls" in _DROP_VARS
        assert "MY24-27_Ls" in _DROP_VARS
        assert "MY28-35_Ls" in _DROP_VARS

    def test_mars_year_variables(self):
        """Mars year variables should be in drop set."""
        assert "MY24-27_MY" in _DROP_VARS
        assert "MY28-35_MY" in _DROP_VARS
