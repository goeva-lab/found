# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
#     language: python
# ---

# %% [markdown]
# # `Pipeline.from_proc_ad` and `HiDDENt`: using preprocessed AnnData objects and hyper-parameter tuning

# %% [markdown]
# we first import all script dependencies and load the provided data in GSE276570 into an anndata object
# %% tags=["hide-input"] mystnb={"code_prompt_show": "show preamble"}
# import dependencies and load data

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from warnings import catch_warnings

import numpy as np
import scanpy as sc

import found
from found import methods as m
from found import pl
from found.adapters import Pipeline
from found.tune import FixPointTuner

RANDOM_STATE = 42
found.set_seed(RANDOM_STATE)

if (pth := Path("../_build/.cache/GSE276570.h5ad")).exists():
    adata = sc.read_h5ad(pth)
else:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE276570&format=file&file="
    with catch_warnings(category=FutureWarning, action="ignore"):
        adata = sc.read_h5ad(BytesIO(urlopen(f"{base_url}GSE276570_endo_object.h5ad").read()))  # pyright: ignore[reportArgumentType]
    adata.obs["injection"] = np.where(adata.obs["Condition"] == 1, "LPC", "saline")
    adata.write_h5ad(pth)

print(adata)

# %% [markdown]
# in the below example, we will utilize pre-computed PCA embeddings.
#
# to do so, we can utilize the `from_proc_ad` static method on `Pipeline`, which will create a pipeline
# that fetches pre-computed embeddings from an `obsm` slot given a predefined key.
# %%
# create a pipeline that uses the above generated dimensionality reduction
algo = Pipeline.from_proc_ad("X_pca", m.log_reg, m.kmeans_bin)

# %% [markdown]
# as we do not know the optimal k for the relabeling task for this dataset, we will
# use the `HiDDENt` entrypoint, which differs from the classic `HiDDEN` entrypoint by requiring
# the user to provide a `Tuner` object, which uses some heuristic to select for a `k` automatically.

# %%

# here we use the provided `FixPointTuner` subclass, which selects for a k s.t. HiDDEN outputs have stabilized relative to smaller ks
tuner = FixPointTuner(4, 3, 0.02)

sel, outs = found.HiDDENt(
    adata,
    "injection",
    "saline",
    algo,
    tuner,
    # ⚠️  the `from_proc_ad` constructor
    #     creates a pipeline which expects
    #     an adata initializing argument
    #     so we must provide it here
    #    ________|
    #    |
    #    V
    adata=adata,
)

# initialize a plotting object
plt = pl.PlotTunerOutput(adata, outs, sel)

# %% [markdown]
# `PlotTunerOutput` provides a plot_scores function which we can
# use to assess the changes in scores across tested k values
# %%
plt.plot_scores()

# %% [markdown]
# we can index into our `PlotTunerOutput` object using
# tested hyperparameters to get a corresponding `PlotHiDDENOutput` object.
#
# here we use this functionality to compare percent relabeling between HiDDEN outputs for k=3 and k=25 (selected value).
# %%
plt_at_start = (
    plt[tuner.start_k][lambda a: a.obs["injection"] == "LPC"]
    .labs_pct("injection", "saline", "animal_id")
    .properties(
        title=f"k = {tuner.start_k}",
    )
)
plt_at_sel = (
    plt[sel][lambda a: a.obs["injection"] == "LPC"]
    .labs_pct("injection", "saline", "animal_id")
    .properties(
        title=f"k = {sel}",
    )
)
plt_at_start | plt_at_sel  # pyright: ignore[reportUnusedExpression]
