# found: R bridge to found python package

this subdirectory provides an R wrapper for the found python package

## disclaimer

very unfinished, more testing is needed before general use can be advised, API is subject to breaking changes under 0ver

## installation

after installing the python library (see [`../README.md`](../README.md)), the R wrapper can be installed via the following command using the [`pak`](https://pak.r-lib.org/#arrow_down-installation) library:

```R
pak::pak("github::goeva-lab/found/R")
```

## documentation

to build documentation a local copy of the repo must first be downloaded

```bash
git clone 'git@github.com:goeva-lab/found' && cd found/R
```

documentation & a manual can then be built using the following R commands

```R
devtools::document()
devtools::build_manual()
```

[`pkgdown`](https://pkgdown.r-lib.org/#installation) can then be used to build a documentation site, which can be built and previewed locally at [localhost:8000](http://localhost:8000) via the following commands (assuming current directory is the [`./R`](.) subdirectory of a clone of this repository):

```bash
R -q -e 'pkgdown::build_site(new_process = FALSE)' # new_process needs to be disabled due to reticulate threading issue
python -m http.server -b localhost -d docs 8000
```
