# found: HiDDEN implementation

this repository provides an implementation of a method for refining case-control labels in sc-RNAseq data, as decribed in [Goeva et al, 2024](https://doi.org/10.1038/s41467-024-53666-8)

## R bridge

an R package providing high-level bindings to the python package is provided, for more documentation on it, please see [`./R/README.md`](./R/README.md).
documentation below is all regarding the python package.

## disclaimer

very unfinished, more testing is needed before general use can be advised, API is subject to breaking changes under 0ver

## TODO

- add more documentation
- add more methods
- host documentation on GH pages
- create GHA to re-build/test/lint/deploy on push to main
- support for more underlying data types (e.g. zarr arrays, HDF5 arrays, etc.)

## installation

[`uv`](https://docs.astral.sh/uv/getting-started/installation/) is used for package management and installation

found can be installed via the following command:

```bash
uv pip install 'git+ssh://git@github.com/goeva-lab/found'
```

## documentation

to build documentation (and run associated notebooks), a local copy of the repo must be downloaded, dev dependencies must be installed, and a kernel based on the configured venv must be installed:

```bash
git clone 'git@github.com:goeva-lab/found' && cd found
uv pip install -e '.[dev]'
uv run -m ipykernel install --user --name found
```

[`sphinx`](https://www.sphinx-doc.org/en/master/) is used to build a documentation website, which can be built and previewed locally at [localhost:8000](http://localhost:8000) via the following command (assuming current directory is a clone of this repository):

```bash
uv run sphinx-autobuild docs docs/_build/html -b dirhtml -n --watch ./src
```
