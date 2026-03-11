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

import pandas as pd
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
    adata.obs["injection"] = pd.Categorical.from_codes(adata.obs["Condition"].eq(1).astype(int), ["saline", "LPC"])
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
tuner_null_dist = NaiveMaxScoreTuner(m.score_null_dist, range(start_k, 31))
tuner_p_hat_dist = NaiveMaxScoreTuner(m.score_dist_diff, range(start_k, 31))

# ⚠️ {py:meth}`~found.adapters.Pipeline.from_proc_ad` creates a pipeline which expects an `adata` argument!
#                                                                            ↓
outs_fix = found.HiDDENt(adata, "injection", "saline", tuner_fix, algo, adata=adata)

# ⚠️ {py:func}`~found.methods.score_null_dist` _requires_ a `pipeline_algo` argument to be provided, so we must inject when calling the pipeline!
#                                                                                                           ↓
outs_null_dist = found.HiDDENt(adata, "injection", "saline", tuner_null_dist, algo, adata=adata, pipeline_algo=algo)

# ⚠️ {py:func}`~found.methods.score_dist_diff` has an _optional_ `score_weight_vsctl` argument but we can override it via injection!
#                                                                                                                ↓
outs_p_hat_dist = found.HiDDENt(adata, "injection", "saline", tuner_p_hat_dist, algo, adata=adata, score_weight_vsctl=0.25)

# initialize corresponding plotting objects
plt_fix = pl.PlotTunerOutput(adata, *outs_fix)
plt_null_dist = pl.PlotTunerOutput(adata, *outs_null_dist)
plt_p_hat_dist = pl.PlotTunerOutput(adata, *outs_p_hat_dist)

# %% [markdown]
# {py:func}`~found.pl.PlotTunerOutput` provides {py:meth}`~found.pl.PlotTunerOutput.score_line` which we can
# use to assess the changes in scores across tested k values
# %%
(
    plt_fix.score_line().properties(title="scores for fix point tuning")
    | plt_null_dist.score_line().properties(title="scores for p_hat distance from null")
    | plt_p_hat_dist.score_line().properties(title="scores for p_hat distance between groups")
).show()

# %% [markdown]
# we can index into our {py:func}`~found.pl.PlotTunerOutput` object using
# tested hyperparameters to get a corresponding {py:func}`~found.pl.PlotTunerOutput` object.
# %%

start = plt_fix[start_k]
fix_point_k = plt_fix[plt_fix.sel]
null_dist_k = plt_null_dist[plt_null_dist.sel]
p_hat_dist_k = plt_p_hat_dist[plt_p_hat_dist.sel]

# %% [markdown]
# here we use this to visualize percent relabeling for the initial tested k, as well as the different ks that were selected for by different tuners
# %%
(
    start.bin_bar("injection", "saline", "animal_id").properties(title=f"k = {start_k}", width=120)
    | fix_point_k.bin_bar("injection", "saline", "animal_id").properties(title=f"k = {plt_fix.sel}", width=120)
    | null_dist_k.bin_bar("injection", "saline", "animal_id").properties(title=f"k = {plt_null_dist.sel}", width=120)
    | p_hat_dist_k.bin_bar("injection", "saline", "animal_id").properties(title=f"k = {plt_p_hat_dist.sel}", width=120)
).show()

# %% [markdown]
# we can also assess p_hat distributions across different ks
# %%
(
    start.reg_vln("injection").properties(title=f"k = {start_k}", width=60)
    | fix_point_k.reg_vln("injection").properties(title=f"k = {plt_fix.sel}", width=60)
    | null_dist_k.reg_vln("injection").properties(title=f"k = {plt_null_dist.sel}", width=60)
    | p_hat_dist_k.reg_vln("injection").properties(title=f"k = {plt_p_hat_dist.sel}", width=60)
).configure_title(anchor="middle").show()
