# found: HiDDEN implementation

this repository provides an implementation of a method for refining case-control labels in single-cell -omics data, as decribed in [Goeva et al, 2024](https://doi.org/10.1038/s41467-024-53666-8)

## disclaimer

API is subject to breaking changes under 0ver, no releases cut so far, breaking changes are pushed to main

## planned future features / TODO

- support for more underlying data types within methods (e.g. zarr arrays, HDF5 arrays as matrix types, etc.)
- native support for multinomial/ordinal condition labels (both in entrypoints and method implementations)
- remove typechecker ignore comments where possible (e.g. via asserts and/or further type hints) & add necessity/safety explainers where missing

## installation

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is used for package management and installation

found can be installed via the following command:

```bash
uv pip install 'git+ssh://git@github.com/goeva-lab/found'
```

alternatively, a docker/OCI image is provided at [ghcr.io/goeva-lab/found](https://github.com/goeva-lab/found/pkgs/container/found) (built according to instructions in [`./Dockerfile`](./Dockerfile) and via the github workflow step [`build_docker`](.github/workflows/workflow.yml#L5-L33)), which provides a debian-based environment with both R and python `found` packages pre-installed.

this image can be downloaded via the following command (replace `docker` w/ `podman`/etc. as needed):

```bash
docker pull 'ghcr.io/goeva-lab/found'
```

## R bridge

an R package providing high-level bindings to the python package is provided, for more documentation on it, please see [`https://goeva-lab.ccbr.utoronto.ca/found/R`](https://goeva-lab.ccbr.utoronto.ca/found/R).
all materials below are regarding the python package.

## documentation

a build of the documentation is publicly available at: [`https://goeva-lab.ccbr.utoronto.ca/found/py`](https://goeva-lab.ccbr.utoronto.ca/found/py).

to build it locally (and run associated notebooks), a copy of the repo must be downloaded, dev dependencies must be installed, and a kernel based on the configured venv must be installed:

```bash
git clone 'git@github.com:goeva-lab/found' && cd found
uv pip install -e '.[dev]'
uv run -m ipykernel install --user --name found
```

[`sphinx`](https://www.sphinx-doc.org/en/master/) is used to build a documentation website, which can be built and previewed locally at [localhost:8000](http://localhost:8000) via the following command (assuming current directory is a clone of this repository):

```bash
uv run sphinx-autobuild './docs' './docs/_build/html' -b 'dirhtml' -n --watch './src'
```

note: this will run and execute all guide vignettes, which can be very computationally expensive.
to avoid doing this, you can pass `-D nb_execution_mode=off` to the above `sphinx-autobuild` command as an additional argument.

## development/contributing

this project uses [`ruff`](https://docs.astral.sh/ruff/installation/) for formatting and [`ty`](https://docs.astral.sh/ty/installation/) for linting.

please ensure that any contributions pass formatting/linting checks accordingly, as determined by `ruff check` and `ty check`.
