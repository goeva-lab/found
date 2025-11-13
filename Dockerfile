from debian:stable-slim as base
run ["apt", "update"]

from base as py_build
run ["mkdir", "-p", "/workdir/found"]
workdir /workdir/found

copy --from=ghcr.io/astral-sh/uv:latest /uv /bin
run ["uv", "venv", ".venv"]

run ["apt", "install", "-y", "rustup"]
run ["rustup", "install", "stable"]
env PATH="/root/.cargo/bin:${PATH}"

run ["apt", "install", "-y", "build-essential"]

copy ./pyproject.toml .
copy ./src ./src
run ["uv", "pip", "install", "."]


from base as py
copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"

run ["python", "-c", "import found"]


from base as R
run ["mkdir", "-p", "/workdir/found"]
workdir /workdir/found

run ["apt", "install", "-y", "r-base-dev"]
run ["apt", "install", "-y", "libcurl4-openssl-dev"]
run ["Rscript", "-e", "install.packages('pak', repos = 'https://cloud.r-project.org/')"]

copy ./R/DESCRIPTION .
copy ./R/R ./R
run ["Rscript", "-e", "pak::local_install_dev_deps()"]

copy --from=py_build /root/.local/share/uv /root/.local/share/uv
copy --from=py_build /workdir/found/.venv /workdir/found/.venv
env PATH="/workdir/found/.venv/bin:${PATH}"
env RETICULATE_PYTHON=/workdir/found/.venv/bin/python
run ["Rscript", "-e", "pak::pak('.')"]

run ["Rscript", "-e", "library(found)"]
