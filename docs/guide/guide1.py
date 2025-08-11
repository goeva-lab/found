# ---
# jupyter:
#   kernelspec:
#     display_name: found
#     name: found
# ---

# %% [markdown]
# # found basics and pipeline customization

# %% [markdown]
# we first load the provided gene expression matrix and
# associated metadata provided in GSE193531 into an anndata object


# %%

from typing import Any

import altair as alt
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import found
import found.methods as m
from found.adapters import Pipeline
from found.types import BoolArr, NumArr

RANDOM_STATE = 42

# %%
# disable SSL certificate due to glitches with GEO SSL certs / pandas cert store
# TODO: find better way to deal with this
gex = pd.read_csv(
    "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE193531&format=file&file=GSE193531_umi-count-matrix.csv.gz",
    index_col=0,
    storage_options={"verify": False},
).T
adata = ad.AnnData(
    gex,
    obs=pd.read_csv(
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE193531&format=file&file=GSE193531_cell-level-metadata.csv.gz",
        storage_options={"verify": False},
    )
    .set_index("index")
    .loc[gex.index],
)

# subset to only disease stages of interest
adata = adata[adata.obs["disease_stage"].isin(["MM", "NBM", "SMM"])].copy()  # pyright: ignore

# use CSR array for counts to improve memory use
adata.X = sp.csr_array(adata.X)

print(adata)

# %% [markdown]
# next, we run the standard HiDDEN pipeline to classify affected cells on:
# normal bone marrow, smoldering multiple myeloma, and multiple myeloma patients

# %%
found.set_seed(RANDOM_STATE)  # set a fixed seed for replicability
algo = Pipeline(m.run_lognorm_pca, m.log_reg, m.kmeans_bin, True)
p_hat, labs = found.HiDDEN(adata, "disease_stage", "NBM", algo, k=30, X=adata.X)

# %% [markdown]
# we define a utility function which plots levels of neoplastic cells per prediction source (e.g. manual vs HiDDEN)


# %%
def plot_res(adata: ad.AnnData, p_hat: NumArr, labs: np.ndarray[tuple[int], Any]) -> alt.Chart | alt.LayerChart:
    eval_df = (
        pd.DataFrame(
            {
                "p_hat": p_hat,
                "prediction": np.where(labs == "NBM", "normal", "neoplastic"),
                "patient": adata.obs["sample_ID"],
                "ground_truth": adata.obs["normal_or_neoplastic"],
            }
        )
        .loc[lambda x: ~x["patient"].str.startswith("NBM")]
        .groupby("patient")[["ground_truth", "prediction"]]
        .agg(lambda x: 1 - (x.eq("normal").sum() / x.size))
        .reset_index()
        .melt(id_vars="patient", value_name="pct_neoplastic", var_name="source")
        .sort_values(by="source")
        .sort_values(by="patient", key=lambda s: pd.to_numeric(s.str.split("-", n=2, expand=True)[1]), kind="stable")
        .sort_values(by="patient", key=lambda s: s.str.split("-", n=2, expand=True)[0], kind="stable", ascending=False)
    )

    c = alt.Chart(eval_df).encode(
        alt.X("patient").title(None).sort(None),
        alt.Y("pct_neoplastic").title("percent neoplastic cells"),
    )

    return c.mark_line(
        strokeDash=(4, 4),
    ).encode(
        alt.Detail("patient"),
    ) + c.mark_point(
        filled=True,
        opacity=0.8,
    ).encode(
        alt.Color("source"),
        alt.Shape("source"),
    )


# %% [markdown]
# evaluating the standard pipeline results, we see strong agreement for the MM samples,
# but a lot less for the SMM samples, with the HiDDEN model consistently predicting a higher amount
# of neoplastic cells as compared to the "ground truth" manual annotations:

# %%
plot_res(adata, p_hat, labs)


# %% [markdown]
# as such, it would be interesting to see if we could modify a component of the default pipeline to improve this.
# to attempt this, we can try to have the binarization step be done in a per-patient fashion.
#
# note: this should not be taken as general advice to follow when troubleshooting found-generated labels, instead proper batch correction methods such as batch-adjusted dimensionality reduction methods should most likely be considered instead.
# however, our interest here is mainly to use this opportunity to demonstrate found's pipeline modification/extension functionality, so we will proceed accordingly.


# %%


# note: this new functionality requires patient metadata, which we make available as a function argument
# importantly, we will then need to "inject" this information into the pipeline when calling invoking it


#                                          patient metadata declared here
#                                                       |
#                                                       V
def per_patient_kmeans_bin(Y: NumArr, V: BoolArr, patient_meta: pd.Series) -> BoolArr:
    out = np.full_like(V, np.nan)

    for patient in patient_meta.unique():
        mask = patient_meta.eq(patient).to_numpy()
        if not V[mask].any():
            continue
        out[mask] = m.kmeans_bin(
            Y[mask],  # pyright: ignore
            V[mask],  # pyright: ignore
        )
    assert not np.isnan(out).any()  # make sure out is properly initialized
    return out.astype(bool)


# we use the `.update` method which returns a new version of the pipeline based off an existing with specified components replaced
algo = algo.update(binr_fn=per_patient_kmeans_bin)

# important! we need to "inject" the patient metadata value into our pipeline
p_hat, labs = found.HiDDEN(
    adata,
    "disease_stage",
    "NBM",
    algo,
    k=30,
    X=adata.X,
    #
    #  introduction of new variables/data into pipeline
    #  is done via extra arguments at invocation point
    #     |
    #     V
    patient_meta=adata.obs["sample_ID"],
)

# %% [markdown]
# assessing the new predictions, as expected, we see almost no difference with our initial results, with anything slightly worse performance on SMM-5:

# %%
plot_res(adata, p_hat, labs)
