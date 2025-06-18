# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
# ---

# %% [markdown]
# # demyelination time course analysis (using processed anndata)


# %%
import warnings
from io import BytesIO
from urllib.request import urlopen

import anndata as ad

# import hdf5plugin  # noqa: F401 - hdf5plugin needed to read h5ad objects

with urlopen("https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE276570&format=file&file=GSE276570_endo_object.h5ad") as f:
    h = BytesIO(f.read())

# silence anndata warning re: adjacency matrix in .uns["neighbors"] being moved to .obsp
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    adata = ad.read_h5ad(h)  # pyright: ignore - read_h5ad type hint too restrictive, BytesIO is also valid

print(adata)
