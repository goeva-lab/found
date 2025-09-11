# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
#     language: python
# ---

# %% [markdown]
# # `HiDDENg` and `HiDDENgt`: grouped entrypoints

# %% [markdown]
# %% [markdown]
# we first import all script dependencies and load the provided data in GSE96583 into an anndata object
# %% tags=["hide-input"] mystnb={"code_prompt_show": "show preamble"}
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
from found.tune import NaiveMaxScoreTuner

RANDOM_STATE = 42
found.set_seed(RANDOM_STATE)

if (pth := Path("../_build/.cache/GSE96583.h5ad")).exists():
    adata = sc.read_h5ad(pth)
else:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/"
    adata = sc.AnnData(
        hstack(
            [
                mmread(BytesIO(decompress(urlopen(url).read()))).tocsc()  # pyright: ignore[reportAttributeAccessIssue]
                for url in [
                    f"{base_url}?acc=GSM2560248&format=file&file=GSM2560248_2.1.mtx.gz",
                    f"{base_url}?acc=GSM2560249&format=file&file=GSM2560249_2.2.mtx.gz",
                ]
            ]
        ).T,
        obs=pd.read_csv(f"{base_url}?acc=GSE96583&format=file&file=GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t")  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
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
# as this is a very common workflow, `found` provides two entrypoints to handle this for the user: `HiDDENg` and `HiDDENgt`, the latter providing hyper-parameter tuning
# %%
phat, labs = found.HiDDENg(
    adata, "stim", "ctrl", "cell", Pipeline(m.run_lognorm_pca, m.logit_reg, m.kmeans_bin), X=adata.X, k=20
)
plt = pl.PlotHiDDENOutput(adata, phat, labs)

# %%
plt[lambda a: a.obs["stim"] == "stim"].labs_pct("stim", "ctrl", "cell")

# %% [markdown]
# we see that megakaryocytes show the most relabeling, so we plot the p_hat values for that cell type specifically
# %%
plt[lambda a: a.obs["cell"] == "Megakaryocytes"].phat_vln("stim", "stim").properties(width=120)

# %% [markdown]
# the above workflow assumes that a hyperparameter k for the pipeline is a) known and b) fixed for all cell types.
# however, this is not always the case, and so we can use the `HiDDENgt` entrypoint to perform tuning combined with factor grouping.

# %%
sel, by_param, by_grp = found.HiDDENgt(
    adata,
    "stim",
    "ctrl",
    "cell",
    Pipeline(m.run_lognorm_pca, m.logit_reg, m.kmeans_bin, cachable_dimr=True),
    NaiveMaxScoreTuner(m.score_phatdiff, range(2, 20, 2)),
    X=adata.X,
)
print(sel)

# %% [markdown]
# due to the permutation of outputs by grouping and hyperparameter selection, `HiDDENgt` returns a set of accessor functions, not raw outputs.
#
# specifically:
#   - the first accessor (here bound to `by_param`), returns `HiDDEN`-style output given a mapping from each group to selected hyperparameters, as well as the score values associated with each group
#   - the second accessor (here bound to `by_grp`), returns `HiDDENt`-style output given a specific group

# %% [markdown]
# here we can plot the re-mixed together outputs of HiDDEN given the optimal hyperparameter for each cell type

# %%
phat, labs, scores = by_param(sel)

pl.PlotHiDDENOutput(adata, phat, labs)[lambda a: a.obs["stim"] == "stim"].labs_pct("stim", "ctrl", "cell")

# %% [markdown]
# we can also explore the evolution of score values for specifically megakaryocytes, and see that setting k to 6 seems to yield optimal results

# %%
mk_out_only = by_grp("Megakaryocytes")
pl.PlotTunerOutput(
    adata[adata.obs["cell"] == "Megakaryocytes"],  # we subset our paired anndata value to only megakaryocytes
    sel["Megakaryocytes"],  # we provided the selected `k` value by indexing the group to choice mapping returned by HiDDENgt
    mk_out_only,
).plot_scores().show()

# %% [markdown]
# finally, it is good to note that `HiDDENg` (and `HiDDENgt`) accept a `grp_specific_args` argument, which provides the ability to inject arguments into the pipeline in a group specific basis.
#
# we can use this here to run `HiDDENg` with different k values per group (using the values selected by `HiDDENgt`).

# %%
phat, labs = found.HiDDENg(
    adata,
    "stim",
    "ctrl",
    "cell",
    Pipeline(m.run_lognorm_pca, m.logit_reg, m.kmeans_bin),
    grp_specific_args={grp: {"k": opt_k} for grp, opt_k in sel.items()},
    X=adata.X,
    k=20,
)

pl.PlotHiDDENOutput(
    adata,
    phat,
    labs,
)[lambda a: a.obs["cell"] == "Megakaryocytes"].phat_vln("stim", "stim").properties(width=120).show()
