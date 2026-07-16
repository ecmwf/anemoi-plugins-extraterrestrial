# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Unit tests for the dynamic distance forcing maker.

Tests the attribute parsing, body resolution, and dispatch logic.
Astropy ephemeris calls are mocked so no network/kernel access is needed.
"""

import datetime as dt
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from anemoi.plugins.extraterrestrial.distance_forcing import _BODY_RADIUS_KM
from anemoi.plugins.extraterrestrial.distance_forcing import _ROTATION_PERIOD_DAYS
from anemoi.plugins.extraterrestrial.distance_forcing import KNOWN_BODIES
from anemoi.plugins.extraterrestrial.distance_forcing import DistanceForcingMaker
from anemoi.plugins.extraterrestrial.distance_forcing import DistanceForcingsSource
from anemoi.plugins.extraterrestrial.distance_forcing import _body_surface_xyz
from anemoi.plugins.extraterrestrial.distance_forcing import _resolve_body
from anemoi.plugins.extraterrestrial.distance_forcing import compute_delta_distance
from anemoi.plugins.extraterrestrial.distance_forcing import compute_distance
from anemoi.plugins.extraterrestrial.distance_forcing import compute_singular_distance

# ---------------------------------------------------------------------------
# Body resolution
# ---------------------------------------------------------------------------


class TestResolveBody:
    """Test body name validation and normalisation."""

    def test_standard_bodies(self):
        for body in ("earth", "moon", "mars", "sun", "jupiter"):
            assert _resolve_body(body) == body

    def test_case_insensitive(self):
        assert _resolve_body("Earth") == "earth"
        assert _resolve_body("MARS") == "mars"

    def test_extended_bodies(self):
        for body in ("phobos", "deimos", "titan", "europa"):
            assert _resolve_body(body) == body

    def test_unknown_body_raises(self):
        with pytest.raises(ValueError, match="Unknown solar system body"):
            _resolve_body("tatooine")

    def test_unknown_body_message_lists_known(self):
        with pytest.raises(ValueError, match="Known bodies"):
            _resolve_body("vulcan")


# ---------------------------------------------------------------------------
# Body surface coordinates
# ---------------------------------------------------------------------------


class TestBodySurfaceXyz:
    """Test spherical coordinate conversion for body surfaces."""

    # At J2000.0 epoch, rotation angle = 0, so body-fixed == GCRS.
    _J2000 = MagicMock()  # mock Time; (time - Time("J2000.0")).jd = 0
    _J2000.__sub__ = lambda self, other: MagicMock(jd=0.0)

    def test_equator_prime_meridian(self):
        """Lat=0, lon=0 at epoch should give (R, 0, 0)."""
        r = _BODY_RADIUS_KM["earth"]
        xyz = _body_surface_xyz("earth", np.array([0.0]), np.array([0.0]), self._J2000)
        assert xyz.shape == (3, 1)
        np.testing.assert_allclose(xyz[:, 0], [r, 0.0, 0.0], atol=1e-10)

    def test_north_pole(self):
        """Lat=90 at epoch should give (0, 0, R) -- poles are rotation-invariant."""
        r = _BODY_RADIUS_KM["mars"]
        xyz = _body_surface_xyz("mars", np.array([90.0]), np.array([0.0]), self._J2000)
        np.testing.assert_allclose(xyz[:, 0], [0.0, 0.0, r], atol=1e-10)

    def test_south_pole(self):
        """Lat=-90 at epoch should give (0, 0, -R)."""
        r = _BODY_RADIUS_KM["moon"]
        xyz = _body_surface_xyz("moon", np.array([-90.0]), np.array([0.0]), self._J2000)
        np.testing.assert_allclose(xyz[:, 0], [0.0, 0.0, -r], atol=1e-10)

    def test_equator_90_east(self):
        """Lat=0, lon=90 at epoch should give (0, R, 0)."""
        r = _BODY_RADIUS_KM["earth"]
        xyz = _body_surface_xyz("earth", np.array([0.0]), np.array([90.0]), self._J2000)
        np.testing.assert_allclose(xyz[:, 0], [0.0, r, 0.0], atol=1e-10)

    def test_radius_is_constant(self):
        """All points should be at distance R from the origin (rotation preserves norm)."""
        r = _BODY_RADIUS_KM["phobos"]
        lats = np.array([0.0, 45.0, -45.0, 90.0, -90.0])
        lons = np.array([0.0, 90.0, 180.0, -90.0, 45.0])
        xyz = _body_surface_xyz("phobos", lats, lons, self._J2000)
        radii = np.linalg.norm(xyz, axis=0)
        np.testing.assert_allclose(radii, r, rtol=1e-10)

    def test_rotation_changes_position(self):
        """Half a rotation period later, equator/prime-meridian should flip sign in x/y."""
        half_period = _ROTATION_PERIOD_DAYS["earth"] / 2.0
        t_half = MagicMock()
        t_half.__sub__ = lambda self, other: MagicMock(jd=half_period)

        xyz_0 = _body_surface_xyz("earth", np.array([0.0]), np.array([0.0]), self._J2000)
        xyz_h = _body_surface_xyz("earth", np.array([0.0]), np.array([0.0]), t_half)

        # After half rotation, (R,0,0) should become (-R,0,0) approximately.
        np.testing.assert_allclose(xyz_h[0, 0], -xyz_0[0, 0], atol=1e-6)
        np.testing.assert_allclose(xyz_h[2, 0], 0.0, atol=1e-10)

    def test_all_bodies_have_radii(self):
        """Every known body should have an entry in _BODY_RADIUS_KM."""
        for body in KNOWN_BODIES:
            assert body in _BODY_RADIUS_KM, f"Missing radius for {body}"
            assert _BODY_RADIUS_KM[body] > 0

    def test_all_bodies_have_rotation_periods(self):
        """Every known body should have a rotation period."""
        for body in KNOWN_BODIES:
            assert body in _ROTATION_PERIOD_DAYS, f"Missing rotation period for {body}"
            assert _ROTATION_PERIOD_DAYS[body] != 0.0


# ---------------------------------------------------------------------------
# Attribute parsing
# ---------------------------------------------------------------------------


class TestDistanceForcingMakerParsing:
    """Test __getattr__ pattern matching and dispatch."""

    def test_parse_distance_pattern(self):
        maker = DistanceForcingMaker()
        parsed = maker._parse_attr("distance_from_earth_to_moon")
        assert parsed == ("distance", "earth", "moon")

    def test_parse_singular_pattern(self):
        maker = DistanceForcingMaker()
        parsed = maker._parse_attr("singular_distance_from_earth_to_moon")
        assert parsed == ("singular", "earth", "moon")

    def test_parse_delta_pattern(self):
        maker = DistanceForcingMaker()
        parsed = maker._parse_attr("delta_distance_from_earth_to_moon")
        assert parsed == ("delta", "earth", "moon")

    def test_parse_different_bodies(self):
        maker = DistanceForcingMaker()
        parsed = maker._parse_attr("distance_from_mars_to_phobos")
        assert parsed == ("distance", "mars", "phobos")

    def test_parse_unrecognised_pattern_returns_none(self):
        maker = DistanceForcingMaker()
        assert maker._parse_attr("velocity_of_earth") is None
        assert maker._parse_attr("distance_earth_moon") is None
        assert maker._parse_attr("something_random") is None

    def test_short_form_returns_none(self):
        """Short forms are not supported (no default_origin)."""
        maker = DistanceForcingMaker()
        assert maker._parse_attr("distance_to_moon") is None


class TestDistanceForcingMakerGetattr:
    """Test __getattr__ dispatches correctly and caches."""

    def test_unknown_attribute_raises_attribute_error(self):
        maker = DistanceForcingMaker()
        with pytest.raises(AttributeError, match="has no attribute"):
            maker.something_unrelated

    def test_unknown_body_raises_value_error(self):
        maker = DistanceForcingMaker()
        with pytest.raises(ValueError, match="Unknown solar system body"):
            maker.distance_from_earth_to_tatooine

    def test_same_origin_target_raises(self):
        maker = DistanceForcingMaker()
        with pytest.raises(AttributeError, match="must differ"):
            maker.distance_from_earth_to_earth

    def test_getattr_returns_callable(self):
        maker = DistanceForcingMaker()
        method = maker.__getattr__("distance_from_earth_to_moon")
        assert callable(method)

    def test_getattr_caches_on_instance(self):
        maker = DistanceForcingMaker()
        _ = maker.__getattr__("distance_from_earth_to_moon")
        # Second access should hit the instance dict, not __getattr__
        assert "distance_from_earth_to_moon" in maker.__dict__

    def test_method_has_name_and_doc(self):
        maker = DistanceForcingMaker()
        method = maker.__getattr__("singular_distance_from_mars_to_phobos")
        assert method.__name__ == "singular_distance_from_mars_to_phobos"
        assert "mars" in method.__doc__.lower()
        assert "phobos" in method.__doc__.lower()


# ---------------------------------------------------------------------------
# SUPPORTED property
# ---------------------------------------------------------------------------


class TestSupported:
    """Test the SUPPORTED property generates valid combinations."""

    def test_supported_is_non_empty(self):
        maker = DistanceForcingMaker()
        assert len(maker.SUPPORTED) > 0

    def test_supported_excludes_self_pairs(self):
        maker = DistanceForcingMaker()
        for name in maker.SUPPORTED:
            parts = name.split("_from_")
            if len(parts) == 2:
                rest = parts[1]
                bodies = rest.split("_to_")
                if len(bodies) == 2:
                    assert bodies[0] != bodies[1]

    def test_supported_contains_all_three_variants(self):
        maker = DistanceForcingMaker()
        supported = maker.SUPPORTED
        assert "distance_from_earth_to_moon" in supported
        assert "singular_distance_from_earth_to_moon" in supported
        assert "delta_distance_from_earth_to_moon" in supported

    def test_supported_contains_mars_phobos(self):
        maker = DistanceForcingMaker()
        assert "distance_from_mars_to_phobos" in maker.SUPPORTED

    def test_supported_count(self):
        """Each body pair (A!=B) should have 3 variants."""
        maker = DistanceForcingMaker()
        n_bodies = len(KNOWN_BODIES)
        n_pairs = n_bodies * (n_bodies - 1)
        assert len(maker.SUPPORTED) == n_pairs * 3


# ---------------------------------------------------------------------------
# Computation with mocked astropy
# ---------------------------------------------------------------------------


def _make_body_mock(x, y, z):
    """Create a mock astropy body with GCRS Cartesian position."""
    body = MagicMock()
    cart = MagicMock()
    cart.x.to.return_value = MagicMock(value=x)
    cart.y.to.return_value = MagicMock(value=y)
    cart.z.to.return_value = MagicMock(value=z)
    body.cartesian = cart
    return body


class TestComputeWithMocks:
    """Test the compute functions with mocked astropy calls."""

    @pytest.fixture
    def mock_astropy(self):
        """Mock astropy so tests run without ephemeris data."""
        # Time is patched to always return this single mock instance,
        # which means both ``Time(date)`` and ``Time("J2000.0")`` yield
        # ``mock_time`` in production code.  We configure ``mock_time -
        # mock_time`` to produce an object whose ``.jd`` is a real
        # scalar so downstream arithmetic (``2*pi*days/period``) works
        # with numpy trig funcs.  Using ``jd = 0.0`` also pins the
        # satellite orbit angle to 0, making Phobos land at
        # ``parent + (semi_major, 0, 0)`` — exactly what the fixture's
        # ``phobos`` body-mock is positioned to match.
        elapsed = MagicMock()
        elapsed.jd = 0.0
        mock_time = MagicMock()
        mock_time.__sub__.return_value = elapsed

        # Earth at origin, Moon at (384400, 0, 0)
        bodies = {
            "earth": _make_body_mock(0.0, 0.0, 0.0),
            "moon": _make_body_mock(384400.0, 0.0, 0.0),
            "mars": _make_body_mock(225000000.0, 0.0, 0.0),
            "phobos": _make_body_mock(225009376.0, 0.0, 0.0),
        }

        with (
            patch(
                "anemoi.plugins.extraterrestrial.distance_forcing.Time",
                return_value=mock_time,
            ),
            patch(
                "anemoi.plugins.extraterrestrial.distance_forcing.get_body",
                side_effect=lambda name, t: bodies[name],
            ),
            patch(
                "anemoi.plugins.extraterrestrial.distance_forcing.solar_system_ephemeris",
            ),
            patch(
                "anemoi.plugins.extraterrestrial.distance_forcing.ASTROPY_AVAILABLE",
                True,
            ),
        ):
            yield bodies

    def test_singular_distance(self, mock_astropy):
        """Singular distance should return a scalar float."""
        result = compute_singular_distance(
            "earth",
            "moon",
            dt.datetime(2026, 1, 1),
        )
        assert isinstance(result, float)
        assert np.isclose(result, 384400.0, rtol=1e-6)

    def test_singular_distance_mars_phobos(self, mock_astropy):
        """Singular distance Mars to Phobos."""
        result = compute_singular_distance(
            "mars",
            "phobos",
            dt.datetime(2026, 1, 1),
        )
        assert isinstance(result, float)
        assert np.isclose(result, 9376.0, rtol=1e-3)

    def test_compute_distance_shape(self, mock_astropy):
        """Per-observer distance should return array of correct shape."""
        lats = np.array([0.0, 45.0, -45.0])
        lons = np.array([0.0, 90.0, 180.0])
        result = compute_distance(
            "earth",
            "moon",
            dt.datetime(2026, 1, 1),
            lats,
            lons,
        )
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_compute_distance_uses_body_radius(self, mock_astropy):
        """Observer positions should use the origin body's radius."""
        # Single observer at equator/prime meridian on Mars.
        # Surface position = (R_mars, 0, 0).
        # Moon target at (384400, 0, 0) in the mock.
        lats = np.array([0.0])
        lons = np.array([0.0])
        result = compute_distance(
            "mars",
            "moon",
            dt.datetime(2026, 1, 1),
            lats,
            lons,
        )
        r_mars = _BODY_RADIUS_KM["mars"]
        expected = abs(384400.0 - r_mars)
        assert np.isclose(result[0], expected, rtol=1e-6)

    def test_delta_distance_non_positive(self, mock_astropy):
        """Delta distance should be <= 0 everywhere."""
        lats = np.array([0.0, 45.0, -45.0])
        lons = np.array([0.0, 90.0, 180.0])
        result = compute_delta_distance(
            "earth",
            "moon",
            dt.datetime(2026, 1, 1),
            lats,
            lons,
        )
        assert result.shape == (3,)
        assert np.all(result <= 0.0 + 1e-10)
        assert np.isclose(np.max(result), 0.0)

    def test_delta_distance_has_variation(self, mock_astropy):
        """Delta distance should show spatial variation."""
        lats = np.array([0.0, 90.0])
        lons = np.array([0.0, 0.0])
        result = compute_delta_distance(
            "earth",
            "moon",
            dt.datetime(2026, 1, 1),
            lats,
            lons,
        )
        # Different observer positions should give different deltas
        assert not np.isclose(result[0], result[1])


# ---------------------------------------------------------------------------
# Integration via __getattr__
# ---------------------------------------------------------------------------


class TestGetAttrIntegration:
    """Test that __getattr__ dispatches to the correct compute function."""

    def test_distance_dispatches_to_compute_distance(self):
        maker = DistanceForcingMaker()
        with patch(
            "anemoi.plugins.extraterrestrial.distance_forcing.compute_distance",
            return_value=np.array([1.0]),
        ) as mock:
            method = maker.distance_from_earth_to_moon
            result = method(dt.datetime(2026, 1, 1), np.array([0.0]), np.array([0.0]))
            mock.assert_called_once()
            assert result[0] == 1.0

    def test_singular_dispatches_correctly(self):
        maker = DistanceForcingMaker()
        with patch(
            "anemoi.plugins.extraterrestrial.distance_forcing.compute_singular_distance",
            return_value=384400.0,
        ) as mock:
            method = maker.singular_distance_from_earth_to_moon
            result = method(dt.datetime(2026, 1, 1))
            mock.assert_called_once()
            assert result == 384400.0

    def test_delta_dispatches_correctly(self):
        maker = DistanceForcingMaker()
        with patch(
            "anemoi.plugins.extraterrestrial.distance_forcing.compute_delta_distance",
            return_value=np.array([-10.0]),
        ) as mock:
            method = maker.delta_distance_from_earth_to_moon
            result = method(dt.datetime(2026, 1, 1), np.array([0.0]), np.array([0.0]))
            mock.assert_called_once()
            assert result[0] == -10.0


# ---------------------------------------------------------------------------
# Astropy not available
# ---------------------------------------------------------------------------


class TestAstropyNotAvailable:
    """Test behaviour when astropy is not installed."""

    def test_singular_raises_import_error(self):
        with patch(
            "anemoi.plugins.extraterrestrial.distance_forcing.ASTROPY_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError, match="astropy"):
                compute_singular_distance(
                    "earth",
                    "moon",
                    dt.datetime(2026, 1, 1),
                )

    def test_distance_raises_import_error(self):
        with patch(
            "anemoi.plugins.extraterrestrial.distance_forcing.ASTROPY_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError, match="astropy"):
                compute_distance(
                    "earth",
                    "moon",
                    dt.datetime(2026, 1, 1),
                    np.array([0.0]),
                    np.array([0.0]),
                )


# ---------------------------------------------------------------------------
# DistanceForcingsSource
# ---------------------------------------------------------------------------


class TestDistanceForcingsSourceInit:
    """Test DistanceForcingsSource initialisation and validation."""

    def test_init_with_template(self):
        ctx = MagicMock()
        template = MagicMock()
        src = DistanceForcingsSource(
            ctx,
            template=template,
            param=["distance_from_earth_to_moon"],
        )
        assert src.template is template
        assert src.param == ["distance_from_earth_to_moon"]

    def test_init_single_param_string(self):
        ctx = MagicMock()
        template = MagicMock()
        src = DistanceForcingsSource(
            ctx,
            template=template,
            param="distance_from_earth_to_moon",
        )
        assert src.param == ["distance_from_earth_to_moon"]

    def test_init_multiple_params(self):
        ctx = MagicMock()
        template = MagicMock()
        src = DistanceForcingsSource(
            ctx,
            template=template,
            param=[
                "distance_from_earth_to_moon",
                "delta_distance_from_earth_to_moon",
                "singular_distance_from_earth_to_moon",
            ],
        )
        assert len(src.param) == 3

    def test_init_unknown_param_raises(self):
        ctx = MagicMock()
        template = MagicMock()
        with pytest.raises(ValueError, match="Unknown distance_forcings param"):
            DistanceForcingsSource(
                ctx,
                template=template,
                param=["velocity_of_earth"],
            )

    def test_init_unknown_body_raises(self):
        ctx = MagicMock()
        template = MagicMock()
        with pytest.raises(ValueError, match="Unknown solar system body"):
            DistanceForcingsSource(
                ctx,
                template=template,
                param=["distance_from_earth_to_tatooine"],
            )
