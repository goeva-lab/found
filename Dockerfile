from debian:stable-slim as base
run ["apt", "update"]

from base as py_build
run ["mkdir", "-p", "/workdir/found"]
workdir /workdir/found

copy --from=ghcr.io/astral-sh/uv:latest /uv /bin
run ["uv", "venv", ".venv", "--managed-python", "-p", "3.14"]

run ["apt", "install", "-y", "rustup"]
run ["rustup", "install", "stable"]
env PATH="/root/.cargo/bin:${PATH}"

run ["apt", "install", "-y", "build-essential"]

copy ./pyproject.toml .
run ["uv", "pip", "compile", "pyproject.toml", "-o", "requirements.txt"]
run ["uv", "pip", "install", "-r", "requirements.txt"]

copy ./src ./src
run ["uv", "pip", "install", "."]
run ["/workdir/found/.venv/bin/python", "-c", "import found"]

from base as py
copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"

from base as R
run ["mkdir", "-p", "/workdir/found"]
workdir /workdir/found

run ["apt", "install", "-y", "r-base-dev", "libcurl4-openssl-dev"]
run ["Rscript", "-e", "install.packages('pak', repos = 'https://cloud.r-project.org/')"]

copy ./R/DESCRIPTION .
run ["Rscript", "-e", "pak::local_install_dev_deps()"]

copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
env RETICULATE_PYTHON="/workdir/found/.venv/bin/python"

copy ./R/R ./R
run ["Rscript", "-e", "pak::pak('local::.')"]

run ["Rscript", "-e", "library(found)"]
