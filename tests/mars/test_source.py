# SPDX-FileCopyrightText: 2026 Anemoi contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Mars source plugin.

These tests do NOT require network access or HuggingFace credentials.
They use synthetic xarray Datasets to test the pure logic functions.
"""

import numpy as np
import pytest
import xarray as xr

from anemoi.plugins.extraterrestrial.mars.source import _DROP_VARS
from anemoi.plugins.extraterrestrial.mars.source import _KNOWN_STORES
from anemoi.plugins.extraterrestrial.mars.source import _LOCAL_ROOT_ENV
from anemoi.plugins.extraterrestrial.mars.source import _RENAME_MACDA
from anemoi.plugins.extraterrestrial.mars.source import CANONICAL_EMARS
from anemoi.plugins.extraterrestrial.mars.source import CANONICAL_MACDA
from anemoi.plugins.extraterrestrial.mars.source import CANONICAL_OPENMARS
from anemoi.plugins.extraterrestrial.mars.source import _merge_openmars_eras
from anemoi.plugins.extraterrestrial.mars.source import _normalise_dataset
from anemoi.plugins.extraterrestrial.mars.source import _open_arco_zarr
from anemoi.plugins.extraterrestrial.mars.source import _process_emars
from anemoi.plugins.extraterrestrial.mars.source import _resolve_local_path
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
        # Local backend: default filename matches the HF store name so a
        # plain download of the HF zarr resolves without extra config.
        assert info["local"]["default"] == "macda_combined.zarr"
        assert info["local"]["default"] == info["hf"]["default"]

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
        assert info["local"]["default"] == "openmars_unified.zarr"
        assert info["local"]["default"] == info["hf"]["default"]

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
        assert info["local"]["default"] == "emars_combined.zarr"
        assert info["local"]["default"] == info["hf"]["default"]

    def test_all_have_required_keys(self):
        """All datasets should have required top-level metadata keys."""
        required_keys = {
            "nlat",
            "nlon",
            "steps_per_sol",
            "frequency_h",
            "hf",
            "earthmover",
            "local",
        }
        for dataset_id, info in _KNOWN_STORES.items():
            assert required_keys.issubset(info.keys()), f"Missing keys in {dataset_id}"

    def test_all_local_backends_have_default(self):
        """Every dataset's local backend must name a default store file."""
        for dataset_id, info in _KNOWN_STORES.items():
            assert "default" in info["local"], f"Missing local default in {dataset_id}"
            assert info["local"]["default"].endswith(".zarr")


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

        def fake_local(canonical, local_path=None):  # pragma: no cover
            called["backend"] = "local"
            return "local-dataset"

        monkeypatch.setattr(src_mod, "_open_hf_zarr", fake_hf)
        monkeypatch.setattr(src_mod, "_open_em_zarr", fake_em)
        monkeypatch.setattr(src_mod, "_open_local_zarr", fake_local)

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

    def test_local_backend_calls_local_opener(self, monkeypatch):
        """backend='local' should call _open_local_zarr with local_path."""
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
            lambda *a, **k: pytest.fail("Earthmover should not be called"),
        )

        def fake_local(canonical, local_path=None):
            called["canonical"] = canonical
            called["local_path"] = local_path
            return "local-dataset"

        monkeypatch.setattr(src_mod, "_open_local_zarr", fake_local)

        result = src_mod._open_arco_zarr(
            "ARCO-MACDA",
            backend="local",
            local_path="/data/mars/macda_combined.zarr",
        )
        assert result == "local-dataset"
        assert called == {
            "canonical": CANONICAL_MACDA,
            "local_path": "/data/mars/macda_combined.zarr",
        }

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


class TestResolveLocalPath:
    """Test _resolve_local_path for the 'local' backend.

    These use real temporary directories to mimic a user who has simply
    downloaded the ``.zarr`` store(s) somewhere on disk and points the
    recipe at them.
    """

    def _make_store(self, base, name):
        """Create a fake .zarr store directory and return its path."""
        store = base / name
        store.mkdir(parents=True)
        # An empty directory is enough for path resolution (we never open it).
        (store / "zarr.json").write_text("{}")
        return store

    def test_explicit_store_path_used_verbatim(self, tmp_path):
        """A path pointing straight at the .zarr store is used as-is."""
        store = self._make_store(tmp_path, "macda_combined.zarr")
        resolved = _resolve_local_path(CANONICAL_MACDA, str(store))
        assert resolved == str(store)

    def test_directory_appends_default_store(self, tmp_path):
        """A directory containing the store resolves to the store path.

        This is the 'downloaded the zarrs independently' case: point at
        the folder and the default filename is appended automatically.
        """
        self._make_store(tmp_path, "macda_combined.zarr")
        resolved = _resolve_local_path(CANONICAL_MACDA, str(tmp_path))
        assert resolved == str(tmp_path / "macda_combined.zarr")

    def test_directory_uses_dataset_specific_default(self, tmp_path):
        """The appended filename matches the canonical dataset."""
        self._make_store(tmp_path, "emars_combined.zarr")
        resolved = _resolve_local_path(CANONICAL_EMARS, str(tmp_path))
        assert resolved.endswith("emars_combined.zarr")

    def test_trailing_slash_directory(self, tmp_path):
        """A directory path with a trailing slash still resolves."""
        self._make_store(tmp_path, "openmars_unified.zarr")
        resolved = _resolve_local_path(CANONICAL_OPENMARS, str(tmp_path) + "/")
        assert resolved.endswith("openmars_unified.zarr")

    def test_env_var_used_when_no_local_path(self, tmp_path, monkeypatch):
        """$ARCO_MARS_LOCAL_ROOT is used when local_path is None."""
        self._make_store(tmp_path, "macda_combined.zarr")
        monkeypatch.setenv(_LOCAL_ROOT_ENV, str(tmp_path))
        resolved = _resolve_local_path(CANONICAL_MACDA, None)
        assert resolved == str(tmp_path / "macda_combined.zarr")

    def test_explicit_path_overrides_env_var(self, tmp_path, monkeypatch):
        """An explicit local_path wins over the environment variable."""
        env_dir = tmp_path / "env"
        arg_dir = tmp_path / "arg"
        self._make_store(env_dir, "macda_combined.zarr")
        self._make_store(arg_dir, "macda_combined.zarr")
        monkeypatch.setenv(_LOCAL_ROOT_ENV, str(env_dir))
        resolved = _resolve_local_path(CANONICAL_MACDA, str(arg_dir))
        assert resolved == str(arg_dir / "macda_combined.zarr")

    def test_missing_path_and_env_raises_value_error(self, tmp_path, monkeypatch):
        """No local_path and no env var is a configuration error."""
        monkeypatch.delenv(_LOCAL_ROOT_ENV, raising=False)
        with pytest.raises(ValueError, match="requires either a 'local_path'"):
            _resolve_local_path(CANONICAL_MACDA, None)

    def test_nonexistent_store_raises_file_not_found(self, tmp_path):
        """Pointing at a store that does not exist is an error."""
        missing = tmp_path / "does_not_exist.zarr"
        with pytest.raises(FileNotFoundError, match="not found at"):
            _resolve_local_path(CANONICAL_MACDA, str(missing))

    def test_directory_without_store_raises_file_not_found(self, tmp_path):
        """A directory missing the expected default store is an error."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="not found at"):
            _resolve_local_path(CANONICAL_MACDA, str(empty))


class TestOpenLocalZarr:
    """Test that _open_local_zarr opens the resolved path and post-processes."""

    def test_opens_resolved_path_and_postprocesses(self, tmp_path, monkeypatch):
        """_open_local_zarr should open the resolved path via xarray.open_zarr."""
        import xarray as xr

        import anemoi.plugins.extraterrestrial.mars.source as src_mod

        store = tmp_path / "macda_combined.zarr"
        store.mkdir()

        opened: dict = {}

        def fake_open_zarr(path, **kwargs):
            opened["path"] = path
            opened["kwargs"] = kwargs
            return xr.Dataset()

        def fake_postprocess(ds, canonical, info):
            opened["canonical"] = canonical
            opened["postprocessed"] = True
            return ds

        monkeypatch.setattr(xr, "open_zarr", fake_open_zarr)
        monkeypatch.setattr(src_mod, "_postprocess_dataset", fake_postprocess)

        src_mod._open_local_zarr(CANONICAL_MACDA, local_path=str(store))

        assert opened["path"] == str(store)
        assert opened["kwargs"]["decode_times"] is False
        assert opened["canonical"] == CANONICAL_MACDA
        assert opened["postprocessed"] is True

    def test_directory_input_resolves_to_store(self, tmp_path, monkeypatch):
        """Passing a directory opens the default store within it."""
        import xarray as xr

        import anemoi.plugins.extraterrestrial.mars.source as src_mod

        store = tmp_path / "emars_combined.zarr"
        store.mkdir()

        opened: dict = {}
        monkeypatch.setattr(
            xr,
            "open_zarr",
            lambda path, **kwargs: opened.setdefault("path", path) or xr.Dataset(),
        )
        monkeypatch.setattr(src_mod, "_postprocess_dataset", lambda ds, *a, **k: ds)

        src_mod._open_local_zarr(CANONICAL_EMARS, local_path=str(tmp_path))
        assert opened["path"] == str(store)


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

    def test_non_time_indexed_variable_present(self):
        """Variables without a ``time`` dimension must not break gap fill.

        Regression test for the EMARS crash. The real EMARS store
        (``ananyo01/ARCO-EMARS/emars_combined.zarr``) contains static
        variables with no leading ``time`` axis:

          * ``anal_mean_Surface_geopotential`` -> dims ``(lat, lon)``,
            shape ``(36, 60)``
          * ``anal_mean_ak`` / ``anal_mean_bk`` -> dims ``(phalf,)``

        The old NaN-slice builder used ``shape[1:]`` and prepended
        ``n_missing``, so for the geopotential it built a ``(n_missing,
        60)`` array with dims ``(lat, lon)`` -> ``lat`` got size
        ``n_missing`` (44 in the report), conflicting with the real
        ``lat`` (36) and raising:
        ``AlignmentError: conflicting dimension sizes: {44, 36}``.

        Uses the real EMARS grid (36 lat x 60 lon, staggered latu/lonv)
        so the reproduction is faithful.
        """
        first_era = np.array([0.0, 1 / 12, 2 / 12])
        second_era = np.array([2.0, 2.0 + 1 / 12, 2.0 + 2 / 12])  # 2 sol gap
        raw_sols = np.concatenate([first_era, second_era])

        lat = np.arange(36)
        lon = np.arange(60)
        latu = np.arange(36)
        lonv = np.arange(60)
        phalf = np.arange(29)

        ds = xr.Dataset(
            {
                # Time-indexed fields on the various (staggered) grids.
                "anal_mean_T": (["time", "lat", "lon"], np.ones((6, 36, 60))),
                "anal_mean_U": (["time", "latu", "lon"], np.ones((6, 36, 60))),
                "anal_mean_V": (["time", "lat", "lonv"], np.ones((6, 36, 60))),
                "anal_mean_ps": (["time", "lat", "lon"], np.ones((6, 36, 60))),
                # Static field (lat, lon) — the exact crash culprit.
                "anal_mean_Surface_geopotential": (
                    ["lat", "lon"],
                    np.ones((36, 60)),
                ),
                # Hybrid-sigma coefficients on the phalf axis (no time).
                "anal_mean_ak": (["phalf"], np.ones(29)),
                "anal_mean_bk": (["phalf"], np.ones(29)),
            },
            coords={
                "time": raw_sols,
                "lat": lat,
                "lon": lon,
                "latu": latu,
                "lonv": lonv,
                "phalf": phalf,
            },
        )

        result = _sort_and_fill_gaps(ds, raw_sols, steps_per_sol=12)

        # Gap filled without error and spatial dims preserved.
        assert len(result.time) > 6
        assert result.sizes["lat"] == 36
        assert result.sizes["lon"] == 60
        assert result.sizes["phalf"] == 29

        # Static variables keep their real shapes (no spurious time axis).
        assert result["anal_mean_Surface_geopotential"].shape == (36, 60)
        assert result["anal_mean_ak"].shape == (29,)
        assert result["anal_mean_bk"].shape == (29,)
        assert "time" not in result["anal_mean_Surface_geopotential"].dims
        assert "time" not in result["anal_mean_ak"].dims

        # Time-carrying vars grew along the time axis, on every grid.
        for v in ("anal_mean_T", "anal_mean_U", "anal_mean_V", "anal_mean_ps"):
            assert result[v].sizes["time"] == len(result.time)

        # Gap region of a time-indexed field is NaN.
        assert np.any(np.isnan(result["anal_mean_T"].values))


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
