from debian:stable-slim as base
run ["mkdir", "-p", "/workdir/found"]

from base as py_build
workdir /workdir/found

run apt -y update &&  \ 
    apt install -y \
    'rustup' 'build-essential' && \
    rm -rf /var/lib/apt/lists/*


# @TODO: remove this if/when unnecessary
# explanation: uv 0.11 seems to have introduced a regression where the GNU_STACK header is set to `X` (as shown by `patchelf --print-execstack`) on the relevant python dylib
# this specifically seems to lead to reticulate being unable to load the python dylib during initialization (can be confirmed by running `patchelf --clear-execstack`)
# as such, pinning uv to 0.10 until this is resolved (no mentions of this issue could be found across other projects as of 2026-03-26)
copy --from=ghcr.io/astral-sh/uv:0.10 /uv /bin

env UV_NO_CACHE=true
env UV_MANAGED_PYTHON=true
env UV_COMPILE_BYTECODE=true
env UV_VENV_RELOCATABLE=true

run ["uv", "venv", "-p", "3.14", ".venv"]

run ["rustup", "install", "stable"]
env PATH="/root/.cargo/bin:${PATH}"

copy ./pyproject.toml .
run ["uv", "pip", "compile", "pyproject.toml", "-o", "requirements.txt"]
run ["uv", "pip", "install", "--strict", "-r", "requirements.txt"]

copy ./src ./src
run ["uv", "pip", "install", "--strict", "."]

from base as py
copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
run ["/workdir/found/.venv/bin/python", "-c", "import found"]

from base as R_build
workdir /workdir/found

run apt -y update &&  \ 
    apt install --no-install-recommends -y \
    'r-base-dev' 'libcurl4-openssl-dev' 'libxml2-dev' 'libuv1-dev' && \
    rm -rf /var/lib/apt/lists/*
run ["Rscript", "-e", "install.packages('pak', repos = sprintf('https://r-lib.github.io/p/pak/stable/%s/%s/%s', .Platform[['pkgType']], R.Version()[['os']], R.Version()[['arch']]))"]
run ["Rscript", "-e", "options(pkg.sysreqs = FALSE); pak::pak('roxygen2')"]

copy ./R/DESCRIPTION .
run ["Rscript", "-e", "options(pkg.sysreqs = FALSE); pak::local_install_deps(lib = '/workdir/.pak/found-lib')"]

copy --from=py /root/.local/share/uv /root/.local/share/uv
copy --from=py /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
env RETICULATE_PYTHON="/workdir/found/.venv/bin/python"

copy ./R/R ./R
run ["Rscript", "-e", ".libPaths('/workdir/.pak/found-lib'); roxygen2::roxygenize()"]
run ["Rscript", "-e", "options(pkg.sysreqs = FALSE); pak::local_install(lib = '/workdir/.pak/found-lib')"]

from base as R

run apt -y update &&  \ 
    apt install --no-install-recommends -y \
    'r-base-core' && \
    rm -rf /var/lib/apt/lists/*

copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
env RETICULATE_PYTHON="/workdir/found/.venv/bin/python"

copy --from=R_build /usr/lib/R/library/ /usr/lib/R/library/
copy --from=R_build /workdir/.pak/found-lib /usr/lib/R/site-library/

run ["Rscript", "-e", "library(found)"]