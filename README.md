# anemoi-plugins-extraterrestrial

<p align="center">
  <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
    <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/sandbox_badge.svg" alt="Maturity Level">
  </a>
  <a href="https://opensource.org/licenses/apache-2-0">
    <img src="https://img.shields.io/badge/Licence-Apache 2.0-blue.svg" alt="Licence">
  </a>
</p>

> \[!IMPORTANT\]
> This software is **Sandbox** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

This is a prototype, experimental plugin set to enable anemoi dataset creation on datasets out of this world.

## Installation

Requires Python >= 3.12.

```bash
# Base install (registers the plugin entry points)
pip install anemoi-plugins-extraterrestrial

# With Mars ARCO backends (HuggingFace-hosted, default)
pip install "anemoi-plugins-extraterrestrial[mars]"

# Additionally with the Earthmover / Arraylake backend
pip install "anemoi-plugins-extraterrestrial[mars,mars-earthmover]"

# With astronomical-distance forcings
pip install "anemoi-plugins-extraterrestrial[distances]"

# From source
git clone https://github.com/ecmwf/anemoi-plugins-extraterrestrial.git
cd anemoi-plugins-extraterrestrial
pip install -e ".[mars]"
```

## Usage

Installing the package registers the following [anemoi-datasets](https://anemoi.readthedocs.io/projects/datasets/) source plugins:

| Entry point | Purpose |
|---|---|
| `arcomars` | Load ARCO-Mars reanalysis (MACDA / OpenMARS / EMARS) from HuggingFace or Earthmover |
| `mars_forcings` | Mars-specific dynamic forcings (sol-of-year, local time, solar longitude, insolation) |
| `distance_forcings` | Astronomical-distance based forcings |

A minimal recipe using the ARCO-Mars source and Mars forcings:

```yaml
input:
  join:
    - arcomars:
        dataset: ARCO-MACDA          # or ARCO-OpenMars / ARCO-EMARS
        # backend: earthmover        # optional; default is hf
    - mars_forcings:
        template: ${input.join.0.arcomars}
        variables:
          - cos_sol_of_year
          - sin_sol_of_year
          - cos_local_time
          - sin_local_time
          - cos_solar_longitude
          - sin_solar_longitude
          - insolation
```

Build a dataset with the standard anemoi-datasets CLI:

```bash
anemoi-datasets create configs/mars/macda.yaml \
    mars-macda-hf-5p0-0024-0035-2h-v1.zarr
anemoi-datasets inspect mars-macda-hf-5p0-0024-0035-2h-v1.zarr
```

## Documentation

Detailed documentation lives close to the code:

- [`configs/mars/README.md`](configs/mars/README.md) — full ARCO-Mars usage guide:
  dataset table, backend selection, variable renaming, forcings design, and
  time-coordinate handling.
- [`configs/mars/macda.yaml`](configs/mars/macda.yaml),
  [`openmars.yaml`](configs/mars/openmars.yaml),
  [`emars.yaml`](configs/mars/emars.yaml) — ready-to-run recipes.
- Module docstrings under
  [`src/anemoi/plugins/extraterrestrial/`](src/anemoi/plugins/extraterrestrial/)
  document each source and forcing (`mars/source.py`, `mars/forcings.py`,
  `mars/revert_time.py`, `distance_forcing.py`).

The `mars-revert-time` CLI (installed with the package) converts a synthetic
Earth-datetime axis back to Mars sols / Ls for downstream analysis.

## Contributing

You can find information about contributing to Anemoi at our [Contribution page](https://anemoi.readthedocs.io/en/latest/contributing/contributing.html).

## License

```
Copyright 2026-, Anemoi Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation
nor does it submit to any jurisdiction.
```
