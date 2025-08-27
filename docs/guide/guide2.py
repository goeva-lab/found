# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
# ---

# %% [markdown]
# # using preprocessed AnnData objects and hyper-parameter tuning


# %%
# import dependencies and download data

from gzip import decompress
from io import BytesIO
from urllib.request import urlopen

import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import hstack

import found
from found import methods as m
from found.adapters import Pipeline
from found.tune import NaiveMinScoreTuner, score_phatdiff

RANDOM_STATE = 42

found.set_seed(RANDOM_STATE)

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

print(adata)

# %% [markdown]
# in the below guide, we will use a non-found implemented processing & dimensionality reduction pipeline

# %%
# generate PCs from analytic pearson residuals
sc.experimental.pp.recipe_pearson_residuals(adata)

# create a pipeline that uses the above generated dimensionality reduction
algo = Pipeline.from_proc_ad("X_pca", m.log_reg, m.kmeans_bin)

# we do now know our optimal k value so we will run a hyperparameter optimization routine
tuner = NaiveMinScoreTuner(score_phatdiff, range(10, 30))

# we use the HiDDENt entry point to utilize hyperparameter optimization
phat, labs = found.HiDDENt(
    adata,
    "stim",
    "ctrl",
    algo,
    tuner,
    # ⚠️ the `from_proc_ad` constructor
    #     creates a pipeline which expects
    #     an adata initializing argument
    #     so we must provide it here
    #    |
    #    |
    #    V
    adata=adata,
)

# we can add the resutling HiDDEN outputs to our anndata object:
adata.obs["hidden_phat"] = phat
adata.obs["hidden_labs"] = labs

print(adata)
