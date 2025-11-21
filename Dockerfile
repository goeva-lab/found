from debian:stable-slim as base
run ["mkdir", "-p", "/workdir/found"]

from base as py_build
workdir /workdir/found

copy --from=ghcr.io/astral-sh/uv:latest /uv /bin
run ["uv", "venv", ".venv", "--managed-python", "-p", "3.13"]

run apt -y update &&  \ 
    apt install -y \
    rustup build-essential && \
    rm -rf /var/lib/apt/lists/*

run ["rustup", "install", "stable"]
env PATH="/root/.cargo/bin:${PATH}"

copy ./pyproject.toml .
run ["uv", "pip", "compile", "pyproject.toml", "-o", "requirements.txt"]
run ["uv", "pip", "install", "-r", "requirements.txt"]

copy ./src ./src
run ["uv", "pip", "install", "."]

from base as py
copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
run ["/workdir/found/.venv/bin/python", "-c", "import found"]

from base as R_build
workdir /workdir/found

run apt -y update &&  \ 
    apt install -y \
    r-base-dev libcurl4-openssl-dev libxml2-dev && \
    rm -rf /var/lib/apt/lists/*
run ["Rscript", "-e", "install.packages(c('pak', 'xml2', 'roxygen2'), repos = 'https://cloud.r-project.org/')"]

copy ./R/DESCRIPTION .
run ["Rscript", "-e", "pak::local_install_deps()"]

copy --from=py /root/.local/share/uv /root/.local/share/uv
copy --from=py /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
env RETICULATE_PYTHON="/workdir/found/.venv/bin/python"

copy ./R/R ./R
run ["Rscript", "-e", "roxygen2::roxygenize()"]
run ["Rscript", "-e", "pak::local_install()"]

run ["Rscript", "-e", "remove.packages(c('pak', 'roxygen2'))"]

from base as R

run apt -y update &&  \ 
    apt install --no-install-recommends -y \
    r-base-core && \
    rm -rf /var/lib/apt/lists/*

copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
env RETICULATE_PYTHON="/workdir/found/.venv/bin/python"

copy --from=R_build /usr/local/lib/R/site-library /usr/local/lib/R/site-library
copy --from=R_build /usr/lib/R/site-library /usr/lib/R/site-library
copy --from=R_build /usr/lib/R/library /usr/lib/R/library

run ["Rscript", "-e", "library(found)"]