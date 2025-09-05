# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
# ---

# %% [markdown]
# # using preprocessed AnnData objects and hyper-parameter tuning

# %% [markdown]
# we first load the provided gene expression matrix and
# associated metadata provided in GSE96583 into an anndata object
# %%
# import dependencies and load data

from gzip import decompress
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import hstack

import found
from found import methods as m
from found import pl
from found.adapters import Pipeline
from found.tune import FixPointTuner

RANDOM_STATE = 42
found.set_seed(RANDOM_STATE)

if (pth := Path("../_build/.cache/GSE96583.h5ad")).exists():
    adata = sc.read_h5ad(pth)
else:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/"
    adata = sc.AnnData(
        hstack(
            [
                mmread(BytesIO(decompress(urlopen(url).read()))).tocsc()  # pyright: ignore
                for url in [
                    f"{base_url}?acc=GSM2560248&format=file&file=GSM2560248_2.1.mtx.gz",
                    f"{base_url}?acc=GSM2560249&format=file&file=GSM2560249_2.2.mtx.gz",
                ]
            ]
        ).T,
        obs=pd.read_csv(f"{base_url}?acc=GSE96583&format=file&file=GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t")  # pyright: ignore
        .reset_index(names="barcode", drop=False)
        .assign(barcode=lambda x: x["stim"] + "_" + x["barcode"].str.extract("([ACTG]+-1)", expand=False))
        .set_index("barcode")[["stim", "cluster", "cell", "multiplets"]],
        var=pd.read_csv(f"{base_url}?acc=GSE96583&format=file&file=GSE96583_batch2.genes.tsv.gz", sep="\t", header=None)
        .set_index(0)
        .rename_axis("ENSEMBL", axis="index")
        .rename({1: "SYMBOL"}, axis="columns"),
    )
    adata.write_h5ad(pth)

print(adata)

# %% [markdown]
# in the below example, we will use a non-found implemented processing & dimensionality reduction pipeline
# %%
# generate PCs from analytic pearson residuals
sc.experimental.pp.recipe_pearson_residuals(adata)

print(adata)
# %%
# create a pipeline that uses the above generated dimensionality reduction
algo = Pipeline.from_proc_ad("X_pca", m.log_reg, m.kmeans_bin)

# we do not know our optimal k value so we will run a hyperparameter optimization routine
tuner = FixPointTuner(4, 3, 0.02)

# we use the HiDDENt entrypoint to utilize hyperparameter optimization
sel, outs = found.HiDDENt(
    adata,
    "stim",
    "ctrl",
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
# tested hyperparameters to plot individual run results
# %%
sel_plt = plt[sel]
sel_plt.labs_pct("stim", "ctrl", "cell")

# %% [markdown]
# the indexed value can be further indexed to plot only a subset of the provided data
# %%
sel_plt[lambda a: a.obs["stim"].eq("stim") & a.obs["cell"].isin(["Megakaryocytes"])].phat_vln("stim")
