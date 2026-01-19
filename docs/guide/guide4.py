# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
#     language: python
# ---

# %% [markdown]
# # pipeline step function creation: a developer's guide

# %% [markdown]
# we first import all script dependencies and load the provided data in GSE96583 into an {py:class}`~anndata.AnnData` object
# %% tags=["hide-input"] mystnb={"code_prompt_show": "show preamble"}
# import dependencies and load data
from gzip import decompress
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import anndata as ad
import numpy as np
import pandas as pd
from pylemur.tl import LEMUR
from scipy.io import mmread
from scipy.sparse import hstack

import found
from found import methods as m
from found import pl
from found.adapters import Pipeline, step_fn
from found.tune import NaiveMaxScoreTuner
from found.types import BoolArr, FloatMtx

RANDOM_STATE = 42
found.set_seed(RANDOM_STATE)

if (pth := Path("../_build/.cache/GSE96583.h5ad")).exists():
    adata = ad.read_h5ad(pth)
else:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/"
    adata = ad.AnnData(
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
# in this vignette, we will explore how to write pipeline components.
# for our example demonstration, we will demonstrate how one could use the LEMUR framework within a HiDDEN pipeline.

# %% [markdown]
# as this vignette is entirely for demonstration purposes, and the LEMUR framework can be computationally intensive, we will also randomly subset our anndata object before proceeding

# %%
# subset to 5% of data
adata = adata[np.random.default_rng(RANDOM_STATE).choice(adata.n_obs, round(adata.n_obs * 0.005), replace=False)].copy()
print(adata)


# %%
# below is a naive implementation of inserting the LEMUR workflow into HiDDEN:
# we run LEMUR, optionally apply the harmony-driven alignment, then return the embedding matrix
def run_lemur(adata: ad.AnnData, k: int, lemur_design: str, lemur_grouping: pd.Series) -> FloatMtx:
    mod = LEMUR(adata, lemur_design, n_embedding=k)
    mod.fit(verbose=False)
    mod.align_with_grouping(lemur_grouping, verbose=False)

    return mod.embedding  # pyright: ignore[reportReturnType]


phat, labs = found.HiDDEN(
    adata,
    "stim",
    "ctrl",
    Pipeline(run_lemur, m.reg_logit, m.bin_kmeans),
    adata=adata,
    k=10,
    lemur_design="~ stim",
    lemur_grouping=adata.obs["cell"],
)

# %% [markdown]
# however, with this implementation of `run_lemur`, further pipeline steps become unable to utilize the LEMUR model.
#
# say that we want to conduct hyperparameter tuning, and during the scoring process, we wish to access the LEMUR counterfactual predictions.
# this would require having the score function access the LEMUR fitted object to generate such counterfactuals.
# however, this object is generated during the dimensionality reduction step, and not returned by `run_lemur`, which returns only the embeddings.
#
# luckily, it is actually possible to have step functions return multiple values!
# this requires the use of the {py:func}`~found.adapters.step_fn` decorator provided in {py:mod}`~found.adapters`.


# %%
# ⚠️ notice the use of the `step_fn` decorator!
#           |
#           V
@step_fn("Z", "lemur_mod")
def run_lemur_w_model_out(adata: ad.AnnData, k: int, lemur_design: str, lemur_grouping: pd.Series) -> tuple[FloatMtx, LEMUR]:
    mod = LEMUR(adata, lemur_design, n_embedding=k)
    mod.fit(verbose=False)
    mod.align_with_grouping(lemur_grouping, verbose=False)

    # ⚠️ we can now return both the embeddings and the model itself
    # ⚠️ important: the order in the tuple has to match the argument provided to the `step_fn` decorator!
    return (
        mod.embedding,  # pyright: ignore[reportReturnType]
        mod,
    )


# ⚠️ we use the "named" lemur_mod output of `run_lemur_w_model_out` in our score function!
#                                 |
#                                 V
def score_lemur_counters(lemur_mod: LEMUR, V: BoolArr, W: BoolArr) -> np.floating:
    only_relab = V & ~W
    only_kept = V & W

    # we compute a score which attempts to:
    # maximize the mean squared difference between LEMUR counterfactuals and actual expression for the HiDDEN-kept cells
    # while minimizing the mean squared difference between LEMUR counterfactuals and actual expression for the HiDDEN-relabeled cells

    relab_dist = (
        lemur_mod.predict(new_condition=lemur_mod.cond(stim="ctrl"))[only_relab]  # pyright: ignore[reportIndexIssue]
        - lemur_mod.data_matrix[only_relab]
    )
    kept_dist = (
        lemur_mod.predict(new_condition=lemur_mod.cond(stim="ctrl"))[only_kept]  # pyright: ignore[reportIndexIssue]
        - lemur_mod.data_matrix[only_kept]
    )

    return np.mean(np.square(kept_dist)) - np.mean(np.square(relab_dist))


sel, out = found.HiDDENt(
    adata,
    "stim",
    "ctrl",
    Pipeline(run_lemur_w_model_out, m.reg_logit, m.bin_kmeans),
    NaiveMaxScoreTuner(score_lemur_counters, range(4, 17, 2)),
    adata=adata,
    lemur_design="~ stim",
    lemur_grouping=adata.obs["cell"],
)

# %% [markdown]
# exploring the HiDDENt outputs, we see that using this metric, we select for `k=10`, with a drop off in score for `k>10`.
# however, given that our data was subset to a very low number of cells for demonstration purposes, these results cannot be interpreted further.

# %%
pl.PlotTunerOutput(adata, sel, out).plot_scores().show()

# %% [markdown]
# this {py:func}`~found.adapters.step_fn` wrapping mechanism is actually used internally during {py:class}`~found.adapters.Pipeline` construction!
# if a provided step function hasn't already been decorated with {py:func}`~found.adapters.step_fn`, {py:class}`~found.adapters.Pipeline` will wrap it in {py:func}`~found.adapters.step_fn`.
# the return value will be named "Z" for {py:class}`~found.adapters.Pipeline```.dimr_fn``, "Y" for {py:class}`~found.adapters.Pipeline```.regr_fn``, and "W" for {py:class}`~found.adapters.Pipeline```.binr_fn``.

# %% [markdown]
# some final notes for developers:
#
# 1) if a function inserted into a pipeline needs dynamic access to *all* pipeline values, this can be done by adding a variable keyword argument (i.e. ``**kwargs`` form).
# for example, this is used by the {py:func}`~found.methods.score_nulldist` function to rerun a pipeline on randomly permuted case labels to approximate a "null" p_hat distribution.
#
# 2) pipeline functions _cannot_ have positional only arguments (either a named positional only argument via the ``arg, /,`` construct or a variable-length positional argument via the ``*args`` construct).
# calling a {py:class}`~found.adapters.Pipeline` where one of the steps is a function with positional-only arguments will raise a {py:exc}`~TypeError` during the validation step executed prior to computation (see {py:func}`found.types.check_step`).
#
# 3) pipeline functions should _always_ use the value of {py:func}`~found.seed.get_seed()` whenever a random seed is necessary, to allow for downstream users to generate fully reproducible results via {py:func}`~found.seed.set_seed()`
