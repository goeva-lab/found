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

to build documentation a local copy of the repo must first be downloaded, then the [`devtools`](https://devtools.r-lib.org/#installation) package can be used to install the package locally and build Rmarkdown documentation.

```bash
git clone 'git@github.com:goeva-lab/found' && cd found/R
R -q -e 'devtools::document(); devtools::install()'
```

the manual can then be built using the following command (again using [`devtools`](https://devtools.r-lib.org/#installation))

```bash
R -q -e 'devtools::build_manual()'
```

a documentation site can be built (using [`pkgdown`](https://pkgdown.r-lib.org/#installation)) and previewed locally at [localhost:8000](http://localhost:8000) via the following command:

```bash
R -q -e 'pkgdown::build_site()' && python -m http.server -b localhost -d docs 8000
```
