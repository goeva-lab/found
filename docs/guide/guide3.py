# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
#     language: python
# ---

# %% [markdown]
# # {py:func}`~found.find.HiDDENg` and {py:func}`~found.find.HiDDENgt`: grouped entrypoints

# %% [markdown]
# %% [markdown]
# we first import all script dependencies and load the provided data in GSE96583 into an {py:class}`~anndata.AnnData` object
# %% tags=["hide-input"] mystnb={"code_prompt_show": "show preamble"}
# import dependencies and load data

from gzip import decompress
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import anndata as ad
import pandas as pd
from scipy.io import mmread
from scipy.sparse import hstack

import found
from found import methods as m
from found import pl
from found.adapters import Pipeline
from found.tune import NaiveMaxScoreTuner

RANDOM_STATE = 42
found.set_seed(RANDOM_STATE)

if (pth := Path("../_build/.cache/GSE96583.h5ad")).exists():
    adata = ad.read_h5ad(pth)
else:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/"
    adata = ad.AnnData(
        hstack(
            [
                mmread(BytesIO(decompress(urlopen(url).read()))).tocsc()
                for url in [
                    f"{base_url}?acc=GSM2560248&format=file&file=GSM2560248_2.1.mtx.gz",
                    f"{base_url}?acc=GSM2560249&format=file&file=GSM2560249_2.2.mtx.gz",
                ]
            ]
        ).T,
        obs=pd.read_csv(f"{base_url}?acc=GSE96583&format=file&file=GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t")
        .reset_index(names="barcode", drop=False)
        .assign(barcode=lambda x: x["stim"] + "_" + x["barcode"].str.extract("([ACTG]+-1)", expand=False))
        .set_index("barcode")[["stim", "cluster", "cell", "multiplets"]],
        var=pd.read_csv(f"{base_url}?acc=GSE96583&format=file&file=GSE96583_batch2.genes.tsv.gz", sep="\t", header=None)
        .set_index(0)
        .rename_axis("ENSEMBL", axis="index")
        .rename({1: "SYMBOL"}, axis="columns"),
    )
    adata = adata[~adata.obs["cell"].isna()].copy()
    adata.write_h5ad(pth)

print(adata)

# %% [markdown]
# it can be of interest to run HiDDEN in a grouped fashion, splitting a dataset
# according to some discrete variable and then running HiDDEN on each subset separately.
#
# as this is a very common workflow, `found` provides two entrypoints to handle this for the user: {py:func}`~found.find.HiDDENg` and {py:func}`~found.find.HiDDENgt`, the latter providing hyper-parameter tuning *and* grouping
# %%
p_hat, labs = found.HiDDENg(adata, "stim", "ctrl", "cell", Pipeline(m.run_pca, m.reg_logit, m.bin_kmeans), X=adata.X, k=20)
plt = pl.PlotHiDDENOutput(adata, p_hat, labs)

# %%
plt.bin_bar("stim", "ctrl", "cell")

# %% [markdown]
# we see that megakaryocytes show the most relabeling, so we plot the p_hat values for that cell type specifically
# %%
plt[lambda a: a.obs["cell"] == "Megakaryocytes"].reg_vln("stim")

# %% [markdown]
# the above workflow assumes that a hyperparameter k for the pipeline is a) known and b) fixed for all cell types.
# however, this is not always the case, and so we can use the {py:func}`~found.find.HiDDENgt` entrypoint to perform tuning combined with factor grouping.

# %%
sel, by_param, by_grp = found.HiDDENgt(
    adata,
    "stim",
    "ctrl",
    "cell",
    NaiveMaxScoreTuner(m.score_ks_diff, range(2, 20, 2)),
    Pipeline(m.run_pca, m.reg_logit, m.bin_kmeans, cachable_dimr=True),
    X=adata.X,
)
print(sel)

# %% [markdown]
# due to the permutation of outputs by grouping and hyperparameter selection, {py:func}`~found.find.HiDDENgt` returns a set of accessor functions, not raw outputs.
#
# specifically:
#   - the first accessor (here bound to `by_param`), returns {py:func}`~found.find.HiDDEN`-style output given a mapping from each group to selected hyperparameters, as well as the score values associated with each group
#   - the second accessor (here bound to `by_grp`), returns {py:func}`~found.find.HiDDENt`-style output given a specific group

# %% [markdown]
# here we can plot the re-mixed together outputs of {py:func}`~found.find.HiDDENgt` given the optimal hyperparameter for each cell type

# %%
p_hat, labs, scores = by_param(sel)

pl.PlotHiDDENOutput(adata, p_hat, labs).bin_bar("stim", "ctrl", "cell")

# %% [markdown]
# we can also explore the evolution of score values for specifically megakaryocytes, and see that setting `k` to `6` seems to yield optimal results

# %%
mk_out_only = by_grp("Megakaryocytes")
pl.PlotTunerOutput(
    adata[adata.obs["cell"] == "Megakaryocytes"],  # we subset our paired anndata value to only megakaryocytes
    sel["Megakaryocytes"],  # we provided the selected `k` value by indexing the group to choice mapping returned by HiDDENgt
    mk_out_only,
).score_line().show()

# %% [markdown]
# finally, it is good to note that {py:func}`~found.find.HiDDENg` (and {py:func}`~found.find.HiDDENgt`) accept a `grp_specific_args` argument, which provides the ability to inject arguments into the pipeline in a group specific basis.
#
# we can use this here to run {py:func}`~found.find.HiDDENg` with different `k` values per group (using the values selected by {py:func}`~found.find.HiDDENgt`).

# %%
p_hat, labs = found.HiDDENg(
    adata,
    "stim",
    "ctrl",
    "cell",
    Pipeline(m.run_pca, m.reg_logit, m.bin_kmeans),
    grp_specific_args={grp: {"k": opt_k} for grp, opt_k in sel.items()},
    X=adata.X,
)

pl.PlotHiDDENOutput(
    adata,
    p_hat,
    labs,
)[lambda a: a.obs["cell"] == "Megakaryocytes"].reg_vln("stim").properties(width=120).show()
