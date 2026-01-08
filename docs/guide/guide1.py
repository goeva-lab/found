# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
#     language: python
# ---

# %% [markdown]
# # `found` and {py:func}`~found.find.HiDDEN`: a whirlwind tour
# %% [markdown]
# we first import all script dependencies and load the provided data in GSE193531 into an {py:class}`~anndata.AnnData` object
# %% tags=["hide-input"] mystnb={"code_prompt_show": "show preamble"}
# import dependencies and load data
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import found
from found import methods as m
from found import pl
from found.adapters import Pipeline
from found.types import BoolArr, NumArr

RANDOM_STATE = 42
found.set_seed(RANDOM_STATE)  # set a fixed seed for replicability

if (pth := Path("../_build/.cache/GSE193531.h5ad")).exists():
    adata = ad.read_h5ad(pth)
else:
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE193531&format=file&file="
    gex = pd.read_csv(f"{base_url}GSE193531_umi-count-matrix.csv.gz", index_col=0).T
    adata = ad.AnnData(
        gex,
        obs=pd.read_csv(f"{base_url}GSE193531_cell-level-metadata.csv.gz").set_index("index").loc[gex.index],
    )
    # subset to only disease stages of interest
    adata = adata[adata.obs["disease_stage"].isin(["MM", "NBM", "SMM"])].copy()  # pyright: ignore[reportArgumentType]

    # use CSR array for counts to improve memory use
    adata.X = sp.csr_array(adata.X)

    # write to cache
    pth.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(pth)

# create a set of labels adjusted using original annotations
adata.obs["disease_stage_gt"] = np.where(adata.obs["normal_or_neoplastic"] == "neoplastic", adata.obs["disease_stage"], "NBM")
adata.obs["sample_ID"] = pd.Categorical(
    adata.obs["sample_ID"],
    categories=sorted(
        sorted(adata.obs["sample_ID"].unique(), key=lambda s: int(s.split("-", 1)[1])),
        key=lambda s: s.split("-", 1)[0],
    ),
    ordered=True,
)

print(adata)

# %% [markdown]
# we run the standard HiDDEN pipeline to classify affected cells on patients from the following groups:
# normal bone marrow, smoldering multiple myeloma, and multiple myeloma
# %%
algo = Pipeline(m.run_pca, m.reg_logit, m.bin_kmeans, True)
p_hat, labs = found.HiDDEN(adata, "disease_stage", "NBM", algo, k=30, X=adata.X)

# %% [markdown]
# to evaluate our pipeline results, we use the provided plotting API
# %%
plt = pl.PlotHiDDENOutput(adata, p_hat, labs)

# %% [markdown]
# evaluating the standard pipeline results, we see strong agreement for the MM samples,
# but a lot less for the SMM samples, with the HiDDEN model consistently predicting a higher amount
# of neoplastic cells as compared to the "ground truth" manual annotations:
# %%
plt.labs_pct("disease_stage_gt", "NBM", "sample_ID").show()

# %% [markdown]
# we can index into the {py:class}`~found.pl.PlotHiDDENOutput` object to only plot a subset of the data (similar to {py:attr}`~pandas.DataFrame.loc` in {py:class}`~pandas.DataFrame`).
# here, we use this to plot p_hat distributions for the three patients where we see the most relabeling, first without splitting by the refined labels, then with splitting:
# %%
subset_plt = plt[lambda a: a.obs["sample_ID"].isin(["SMM-3", "SMM-8", "SMM-10"])]
(
    subset_plt.phat_vln("sample_ID", split_mode=False).properties(width=120)
    | subset_plt.phat_vln("sample_ID").properties(width=120)
).show()

# %% [markdown]
# it might be of interest to ask if per-patient batch effects have serious effects on HiDDEN outputs, and if accounting for them could increase the accuracy/sensitivity of our outputs.
# to attempt this, we can try to have the binarization step be done in a per-patient fashion.
#
# note: this should not be taken as general advice to follow when troubleshooting found-generated labels, instead proper batch correction methods such as batch-adjusted dimensionality reduction should most likely be considered instead.
# however, our interest here is mainly to use this opportunity to demonstrate found's pipeline modification/extension functionality, so we will proceed accordingly.
# %%
# note: this new functionality requires patient metadata, which we make available as a function argument
# importantly, we will then need to "inject" this information into the pipeline when calling invoking it


#                                          patient metadata declared here
#                                                       |
#                                                       V
def per_patient_kmeans_bin(Y: NumArr, V: BoolArr, patient_meta: pd.Series) -> BoolArr:
    out = V.copy()

    for patient in patient_meta.unique():
        mask = patient_meta.eq(patient).to_numpy()
        if not V[mask].any():
            continue
        out[mask] = m.bin_kmeans(
            Y[mask],  # pyright: ignore[reportArgumentType]
            V[mask],  # pyright: ignore[reportArgumentType]
        )

    return out.astype(bool)


# we use the `.update` method which returns a new version of the pipeline based off an existing with specified components replaced
algo = algo.update(binr_fn=per_patient_kmeans_bin)

# ⚠️ important: we need to "inject" the patient metadata value into our pipeline
p_hat, labs = found.HiDDEN(
    adata,
    "disease_stage",
    "NBM",
    algo,
    k=30,
    X=adata.X,
    #  ⚠️ introduction of new variables/data into pipeline
    #  ⚠️ is done via extra arguments at invocation point
    #           |
    #           V
    patient_meta=adata.obs["sample_ID"],
)
plt = pl.PlotHiDDENOutput(adata, p_hat, labs)

# %% [markdown]
# assessing the new predictions, as expected, we see almost no difference with our initial results, with anything slightly worse performance on SMM-5:
# %%
plt.labs_pct("disease_stage_gt", "NBM", "sample_ID").show()
