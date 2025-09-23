# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
#     language: python
# ---

# %% [markdown]
# # {py:meth}`~found.adapters.Pipeline.from_proc_ad` and {py:func}`~found.find.HiDDENt`: using preprocessed {py:class}`~anndata.AnnData` objects and hyper-parameter tuning

# %% [markdown]
# we first import all script dependencies and load the provided data in GSE276570 into an {py:class}`~anndata.AnnData` object
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
from found.tune import FixPointTuner, NaiveMaxScoreTuner

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
# to do so, we can utilize the {py:meth}`~found.adapters.Pipeline.from_proc_ad` static method on {py:class}`~found.adapters.Pipeline`, which will create a pipeline
# that fetches pre-computed embeddings from an `obsm` slot given a predefined key.
# %%
# create a pipeline that uses the above generated dimensionality reduction
algo = Pipeline.from_proc_ad("X_pca", m.reg_logit, m.bin_kmeans)

# %% [markdown]
# as we do not know the optimal k for the relabeling task for this dataset, we will
# use the {py:func}`~found.find.HiDDENt` entrypoint, which differs from the classic {py:func}`~found.find.HiDDEN` entrypoint by requiring
# the user to provide a {py:class}`~found.tune.Tuner` object, which attempts to use some heuristic to select for pipeline hyperparameters automatically.

# %%

# here we initialize a variety of tuners, and will compare their results
start_k = 3
tuner_fix = FixPointTuner(4, start_k, 0.02)
tuner_nulldist = NaiveMaxScoreTuner(m.score_nulldist, range(start_k, 31))
tuner_phatdist = NaiveMaxScoreTuner(m.score_phatdiff_dist, range(start_k, 31))

# ⚠️  the `from_proc_ad` constructor creates a pipeline which expects an `adata` argument!
# ⚠️            |_____________________________________________________________
# ⚠️                                                                         |
# ⚠️                                                                         V
outs_fix = found.HiDDENt(adata, "injection", "saline", algo, tuner_fix, adata=adata)

# ⚠️  the `score_nulldist` function _requires_ a `pipeline_algo` argument to be provided, so we must inject that as well
# ⚠️            |____________________________________________________________________________________________
# ⚠️                                                                                                        |
# ⚠️                                                                                                        V
outs_nulldist = found.HiDDENt(adata, "injection", "saline", algo, tuner_nulldist, adata=adata, pipeline_algo=algo)

# ⚠️  the `score_phatdiff_emd` has an _optional_ `score_weight_vsctl` but we can override it via injection as well
# ⚠️            |_________________________________________________________________________________________________
# ⚠️                                                                                                             |
# ⚠️                                                                                                             V
outs_phatdist = found.HiDDENt(adata, "injection", "saline", algo, tuner_phatdist, adata=adata, score_weight_vsctl=0.25)

# initialize corresponding plotting objects
plt_fix = pl.PlotTunerOutput(adata, *outs_fix)
plt_nulldist = pl.PlotTunerOutput(adata, *outs_nulldist)
plt_phatdist = pl.PlotTunerOutput(adata, *outs_phatdist)

# %% [markdown]
# {py:func}`~found.pl.PlotTunerOutput` provides a plot_scores function which we can
# use to assess the changes in scores across tested k values
# %%
(
    plt_fix.plot_scores().properties(title="scores for fix point tuning")
    | plt_nulldist.plot_scores().properties(title="scores for p_hat distance from null")
    | plt_phatdist.plot_scores().properties(title="scores for p_hat distance between groups")
).show()

# %% [markdown]
# we can index into our {py:func}`~found.pl.PlotTunerOutput` object using
# tested hyperparameters to get a corresponding {py:func}`~found.pl.PlotTunerOutput` object.
# %%

case_mask = adata.obs["injection"] == "LPC"
start = plt_fix[start_k]
fixk = plt_fix[plt_fix.sel]
nulldistk = plt_nulldist[plt_nulldist.sel]
phatdistk = plt_phatdist[plt_phatdist.sel]

# %% [markdown]
# here we use this to visualize percent relabeling for the initial tested k, as well as the different ks that were selected for by different tuners
# %%
(
    start[case_mask].labs_pct("injection", "saline", "animal_id").properties(title=f"k = {start_k}", width=120)
    | fixk[case_mask].labs_pct("injection", "saline", "animal_id").properties(title=f"k = {plt_fix.sel}", width=120)
    | nulldistk[case_mask].labs_pct("injection", "saline", "animal_id").properties(title=f"k = {plt_nulldist.sel}", width=120)
    | phatdistk[case_mask].labs_pct("injection", "saline", "animal_id").properties(title=f"k = {plt_phatdist.sel}", width=120)
).show()

# %% [markdown]
# we can also assess phat distributions across different ks
# %%
(
    start.phat_vln("injection").properties(title=f"k = {start_k}", width=60)
    | fixk.phat_vln("injection").properties(title=f"k = {plt_fix.sel}", width=60)
    | nulldistk.phat_vln("injection").properties(title=f"k = {plt_nulldist.sel}", width=60)
    | phatdistk.phat_vln("injection").properties(title=f"k = {plt_phatdist.sel}", width=60)
).configure_title(anchor="middle").show()
