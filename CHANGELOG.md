# Changelog

## [0.1.3](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/compare/0.1.2...0.1.3) (2026-08-05)


### Bug Fixes

* **mars:** Handle non-time-indexed variables in EMARS sol-gap fill ([#40](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/issues/40)) ([3af4773](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/3af477378a36eb652a18cab9bdcc111be9d277f3))
* **mars:** Reconstruct continuous sol axis for EMARS gap detection ([#42](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/issues/42)) ([bc06d8a](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/bc06d8a28b124ea8975713280374cb300e0c8aa3))

## [0.1.2](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/compare/0.1.1...0.1.2) (2026-08-04)


### Features

* **mars:** Add local backend to arcomars source ([#38](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/issues/38)) ([c12c760](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/c12c760450e837c0b78d99d31a2336d97d5b990c))

## [0.1.1](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/compare/0.1.0...0.1.1) (2026-07-21)


### Bug Fixes

* **packaging:** Declare readme + metadata so PyPI renders the description ([#27](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/issues/27)) ([50de20e](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/50de20e047b80486083905ff12a7c50c1a170189))

## 0.1.0 (2026-07-20)


### Features

* Add arbitary distance forcings ([ef59ead](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/ef59eadabba72cdecbca4ad3223873d9d1429947))
* Add EMARS support, solar_longitude forcing, per-dataset grid/frequency ([861a883](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/861a883e4fdc4987285b7c2932673d3f992825e6))
* Mars ([34c545b](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/34c545b83d38c762822f951056596fb092f6f153))
* **mars:** Add earthmover backend to arcomars source ([fe9bde5](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/fe9bde5b5109a988d24cd1166721748d51d31d64))
* Regridding matrices + configs ([#1](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/issues/1)) ([a0125c8](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/a0125c8a72d9bd7f892241f4087ac58637c99b68))
* Sort+dedup+gap-fill for sol timeline, tests, combined config ([63e5830](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/63e5830841b06dcdd618d665a9ef141834e5d030))


### Bug Fixes

* **ci:** Dedupe dependabot github-actions directory config ([#23](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/issues/23)) ([3b833fd](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/3b833fd8a01e17302c051e2704f6af1088288ce1))
* **config:** Add missing block for MY27-28 gap in OpenMars configs ([5d62793](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/5d62793f7e25956ae13a328693586018191452a2))
* **config:** Correct combined dataset end date and remove stale missing block ([aceacf0](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/aceacf0114bfea973d093984ba4d86ff36bc820b))
* **config:** Correct end dates for openmars and combined ([75f0e92](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/75f0e92338785aa809dda5024e7734936c28e871))
* **config:** Correct openmars and combined end dates, remove stale missing blocks ([738f394](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/738f394a7d840e6e840e33483a9bf0e10db3ae06))
* **config:** Correct openmars end date and remove stale missing block ([91655ed](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/91655ed8ec10b0eb357c3e35a99568fc67550cc0))
* Correct EMARS dataset name (1h frequency, 5p0x6p0 resolution) ([1ce8bc5](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/1ce8bc54948cd4e78cae7a7a84d1d1b732a33480))
* **distance_forcing:** Compute_delta_distance subtracted np.min, which ([fe9bde5](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/fe9bde5b5109a988d24cd1166721748d51d31d64))
* Restore NaN gap insertion, add resolution monkey-patch, update configs ([5b2fc8e](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/5b2fc8e78ed8962d13b723b208fa8e8fdf0c6317))
* **tests:** Mock_astropy fixture now provides Time.__sub__ -&gt; obj.jd=0.0 ([fe9bde5](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/fe9bde5b5109a988d24cd1166721748d51d31d64))
* Use dask arrays for NaN gap fill, explicit time coord load ([92be1c0](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/92be1c0a9804be755ea59ad71dd0d9c9d71ea82a))
* Use new_field_from_numpy for forcings data, dual templates for metadata ([2c3215f](https://github.com/ecmwf/anemoi-plugins-extraterrestrial/commit/2c3215f7a9d850720d492acd6a6d9230aa018747))

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Please add your functional changes to the appropriate section in the PR.
Keep it human-readable, your future self will thank you!
