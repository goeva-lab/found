from typing import Any, Iterable

import anndata as ad
import numpy as np

from .adapters import Pipeline, heuristic_loop, strip_generic
from .pipelines import LogNormPCALogRegKMeansKSScore
from .types import MatrixLike, NumArr


def find(
    x: ad.AnnData,
    cond_col: str,
    control_val: Any,
    algo: Pipeline = LogNormPCALogRegKMeansKSScore,
    k_range: Iterable[int] = tuple(range(10, 50)),
    layer: str | None = None,
    **kwargs,
) -> tuple[NumArr, np.ndarray[tuple[int], Any]]:
    """
    runs HiDDEN on a given AnnData object, trying to optimize for a specific set of dimensions specified by k_range.

    :param adata: input AnnData object
    :param cond_col: string indicating obs column in adata representing condition value
    :param control_val: value representing the control condition in the provided condition column
    :param algo: algorithm pipeline (expected to use parameters ``X`` for counts matrix, ``V`` as original condition annotation, and ``k`` as number of dimensions to reduce to)
    :param k_range: range of dimensionality reduction dimensions to iterate over when trying to find best value
    :param layer: layer to pull data from if not using X matrix for counts
    :param kwargs: additional variables to pass into pipeline
    :return: 2-tuple consisting of:

        - 1-d array of prediction outputs by model
        - binarized labels from prediction values
    """
    X = x.X if layer is None else x.layers[layer]

    if not isinstance(X, strip_generic(MatrixLike)):
        raise ValueError(f"type of expression matrix: {type(X)} is not currently supported")
    y_hat, new_ann, _ = heuristic_loop(algo, k_range, X=X, V=(x.obs[cond_col] != control_val).to_numpy(), **kwargs)
    new_ann = np.where(new_ann, x.obs[cond_col], control_val)

    return (
        y_hat,
        new_ann,  # pyright: ignore
        # ignore NECESSITY - np.where broadcasting does
        # not maintain array size in type information
        # above assert validates this at runtime
    )
