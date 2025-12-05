from collections.abc import Callable
from numbers import Integral
from typing import Any, Protocol, Self, runtime_checkable, Literal
from warnings import catch_warnings

import numpy as np
import scipy.sparse as sp
from scipy.special import kl_div
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import sparsefuncs

from .adapters import _INTERNAL_WRAP_ATTR_NAME, Pipeline, out_to_dict, step_fn, wcall
from .seed import get_seed
from .types import BoolArr, FloatMtx, MatrixLike, NumArr, NumericScalar


def mult_preserve_type[T: MatrixLike](lhs: T, rhs: np.ndarray) -> T:
    o = lhs * rhs

    if isinstance(lhs, sp.csr_array):
        o = o.tocsr()  # pyright: ignore[reportAttributeAccessIssue]
    elif isinstance(lhs, sp.csc_array):
        o = o.tocsc()  # pyright: ignore[reportAttributeAccessIssue]

    return o  # pyright: ignore[reportReturnType]


def scale_rs[T: MatrixLike](X: T) -> T:
    """
    returns scaled version of input matrix with row-wise sums of 1

    :param X: input matrix
    :return: scaled matrix
    """
    per_cell_sum = X.sum(axis=1)
    avg_counts_per_cell = per_cell_sum.mean()
    size_fact = per_cell_sum / avg_counts_per_cell

    # silence warnings about reciprocal for zero, filled with zeros elsewhere
    with catch_warnings(action="ignore", category=RuntimeWarning):
        recip = np.reciprocal(size_fact)
    recip = np.where(size_fact > 0, recip, 0.0)

    return mult_preserve_type(X, recip[:, np.newaxis])  # pyright: ignore[reportReturnType]
    # ignore NECESSITY - np.where broadcasting does not maintain array size in type information


def scale_sd[T: MatrixLike](X: T) -> T:
    """
    returns scaled version of input matrix with column-wise standard deviation of 1

    :param X: input matrix
    :return: scaled matrix
    """

    if isinstance(X, np.ndarray):
        stdev = np.std(X, axis=0)
    else:
        _, var = sparsefuncs.mean_variance_axis(X, axis=0)  # pyright: ignore[reportArgumentType, reportAssignmentType, reportGeneralTypeIssues]
        # ignore NECESSITY - sparsefuncs.mean_variance_axis return type set to
        # Unknown, however type is documented to be 2-tuple given these arguments
        stdev = np.sqrt(var)

    # silence warnings about reciprocal for zero, filled with zeros elsewhere
    with catch_warnings(action="ignore", category=RuntimeWarning):
        recip = np.reciprocal(stdev)
    recip = np.where(stdev > 0, recip, 0.0)

    return mult_preserve_type(X, recip[np.newaxis, :])  # pyright: ignore[reportReturnType]
    # ignore NECESSITY - np.where broadcasting does not maintain array size in type information


def norm_log1p(X: MatrixLike | sp.csc_matrix | sp.csr_matrix) -> MatrixLike:
    """
    implements a size factor scaled log1p transform as recommended in `Ahlmann-Eltze et al. <https://doi.org/10.1038/s41592-023-01814-1>`_.

    :param X: input cell by gene matrix
    :return: ``log((X / s) + 1)`` where ``s`` is the per-cell size factors of ``X``
    """

    # if dealing with spmatrix types, convert to sparray equivalents
    if isinstance(X, sp.csr_matrix):
        X = sp.csr_array(X)
    if isinstance(X, sp.csc_matrix):
        X = sp.csc_array(X)

    X = scale_rs(X)

    if isinstance(X, np.ndarray):
        X = np.log1p(X)  # pyright: ignore[reportAssignmentType]
        # ignore NECESSITY - log1p method type hint does not preserve shape type hint
    else:
        X = X.tocsr().log1p()  # pyright: ignore[reportAttributeAccessIssue]
        # ignore NECESSITY - pyright can't detect log1p method on sparse matrix types
        # because method is added dynamically, as observed here:
        # https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/_data.py#L138
        # https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/_base.py#L52

    # make sure we're not introducing sp.spmatrix type further into the pipeline
    assert not isinstance(X, sp.spmatrix)

    return X


def run_pca(N: MatrixLike, k: int, scale: bool = True) -> FloatMtx:
    """
    runs PCA to provide PC-space coordinates for input matrix, first applying a centering and scaling transform.

    :param N: cell by gene matrix
    :param k: dimensionality of the PCA to run
    :return: cell by k matrix representing cells in PC-space
    """

    # return PCA output
    # centering is not needed since arpack solver centers pre-PCA
    return PCA(k, svd_solver="arpack", random_state=get_seed()).fit_transform(
        scale_sd(N) if scale else N  # pyright: ignore[reportArgumentType]
    )
    # ignore NECESSITY - sparray also works for PCA, but is not documented


def run_lognorm_pca(X: MatrixLike | sp.csc_matrix | sp.csr_matrix, k: int, scale: bool = True) -> FloatMtx:
    """
    runs PCA to provide PC-space coordinates for each cell, first applying a log1p w/ scaling transform, then centering and scaling.

    :param X: cell by gene matrix
    :param k: dimensionality of the PCA to run
    :return: cell by k matrix representing cells in PC-space
    """
    return run_pca(norm_log1p(X), k, scale)


@step_fn("Z", "NMF_p")
def run_nmf(
    X: MatrixLike,
    k: int,
    sf_obs_scale: bool = True,
    sd_var_scale: bool = True,
    nmf_lreg: tuple[float, float] = (0.0, 0.0),
    nmf_l1l2ratio: float = 0.0,
) -> tuple[FloatMtx, FloatMtx]:
    """
    runs NMF (decomposition of ``X`` into ``w @ h``) as implemented in :py:class:`~sklearn.decomposition.NMF` to provide NMF-cell-by-k-space coordinates for each cell.

    note:

    - ``w`` refers to the cell by k matrix
    - ``h`` refers to the k by gene matrix

    see :py:class:`~sklearn.decomposition.NMF` documentation for more details on meaning of arguments.

    :param X: cell by gene matrix
    :param k: k, specifying number of NMF components/programs
    :param sf_obs_scale: should row-wise size factor scaling be applied
    :param sd_var_scale: should column-wise standard deviation scaling be applied
    :param nmf_lreg: tuple of penalty terms for w and h matrices, respectively (0 meaning no regularization is applied)
    :param nmf_l1l2ratio: regularization mixing parameter, indicating weight of l1 vs l2 penalty (0 means only l2 penalty, 1 means only l1 penalty)
    :return: 2-tuple of:

        - cell by k matrix representing cells in NMF-cell-by-k-space (e.g. ``w``)
        - gene by k matrix representing NMF-computed gene programs (e.g. ``h^T``)
    """

    if sf_obs_scale:
        X = scale_rs(X)
    if sd_var_scale:
        X = scale_sd(X)

    m = NMF(
        k,  # pyright: ignore[reportArgumentType]
        random_state=get_seed(),
        alpha_W=nmf_lreg[0],
        alpha_H=nmf_lreg[1],  # pyright: ignore[reportArgumentType]
        l1_ratio=nmf_l1l2ratio,
    )
    z = m.fit_transform(X)  # pyright: ignore[reportArgumentType]

    return z, m.components_.T


class HasFitBinClass(Protocol):
    # @TODO: remove `| Any` escape hatches

    # Any required since `GaussianProcessClassifier.classes_` attribute is set to be ArrayLike/Buffer, not just ndarray
    classes_: np.ndarray[tuple[int], np.dtype[np.bool]] | Any
    # Any required since optional arguments on fit method are not properly interpreted
    fit: Callable[[FloatMtx, BoolArr], Self] | Any
    # Any required due to SVC predict_proba type hint being unspecific
    predict_proba: Callable[[FloatMtx], np.ndarray[tuple[int, int], np.dtype[np.floating]]] | Any


def sklearn_wrap[T: HasFitBinClass](Z: FloatMtx, V: np.ndarray, model: T) -> tuple[NumArr | FloatMtx, T]:
    is_binary = set(np.unique(V)) == {False, True}
    
    model = model.fit(Z, V)
    if is_binary:
        return (
            # classes_ is generated by np.unique, which returns the classes
            # in sorted order, so True should always be the second class
            # however we still perform the `.index` operation as a sanity check
            model.predict_proba(Z)[:, model.classes_.tolist().index(True)],  # pyright: ignore[reportReturnType]
            # ignore NECESSITY - `mode.predict_proba` will return a 2-d array
            # so indexing with [:, int] will return yield a 1-d array
            model,
        )
    else:
        return model.predict_proba(Z), model


@step_fn("Y", "model")
def reg_logit(
    Z: FloatMtx,
    V: BoolArr,
    logit_args: dict[str, Any] | None = None,
) -> tuple[NumArr | FloatMtx, LogisticRegression]:
    """
    runs a logistic regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param logit_args: additional arguments to pass to :py:class:`~sklearn.linear_model.LogisticRegression` (defaults: ``max_iter``: ``100``, ``solver``: ``"newton-cg"``, ``penalty``: ``None``)
    :return: 2-tuple of:

        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """
    return sklearn_wrap(
        Z,
        V,
        LogisticRegression(
            random_state=get_seed(),
            **({"solver": "newton-cg", "max_iter": 100, "penalty": None} | (logit_args or {})),
        ),
    )


@step_fn("Y", "model")
def reg_svm(
    Z: FloatMtx,
    V: BoolArr,
    svm_args: dict[str, Any] | None = None,
) -> tuple[NumArr | FloatMtx, SVC]:
    """
    runs a support vector machine classifier -based regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param svm_args: additional arguments to pass to :py:class:`~sklearn.svm.SVC`
    :return: 2-tuple of:

        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """
    return sklearn_wrap(
        Z,
        V,
        SVC(
            probability=True,
            random_state=get_seed(),
            **(svm_args or {}),
        ),
    )


@step_fn("Y", "model")
def reg_gp(
    Z: FloatMtx,
    V: BoolArr,
    gp_args: dict[str, Any] | None = None,
) -> tuple[NumArr | FloatMtx, GaussianProcessClassifier]:
    """
    runs a Gaussian process classifier -based regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param gp_args: additional arguments to pass to :py:class:`~sklearn.gaussian_process.GaussianProcessClassifier`
    :return: 2-tuple of:

        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """

    return sklearn_wrap(
        Z,
        V,
        GaussianProcessClassifier(
            random_state=get_seed(),
            **(gp_args or {}),
        ),
    )


@step_fn("Y", "model")
def reg_rf(
    Z: FloatMtx,
    V: BoolArr,
    rf_args: dict[str, Any] | None = None,
) -> tuple[NumArr | FloatMtx, RandomForestClassifier]:
    """
    runs a random forest classifier -based regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param rf_args: additional arguments to pass to :py:class:`~sklearn.ensemble.RandomForestClassifier`
    :return: 2-tuple of:

        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """

    return sklearn_wrap(
        Z,
        V,
        RandomForestClassifier(
            random_state=get_seed(),
            **(rf_args or {}),
        ),
    )

@step_fn("Y", "model")
def reg_nc(
    Z: FloatMtx,
    V: np.ndarray,
    nc_args: dict[str, Any] | None = None,
) -> tuple[NumArr | FloatMtx, NearestCentroid]:
    """
    runs a Nearest Centroid classifier -based regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d array of condition labels
    :param svm_args: additional arguments to pass to :py:class:`~sklearn.neighbors.NearestCentroid`
    :return: 2-tuple of:
        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """
    return sklearn_wrap(
        Z,
        V,
        NearestCentroid(
            **(nc_args or {}),
        ),
    )


@step_fn("Y", "model")
def reg_rn(
    Z: FloatMtx,
    V: np.ndarray,
    nc_args: dict[str, Any] | None = None,
) -> tuple[NumArr | FloatMtx, RadiusNeighborsClassifier]:
    """
    runs a Radius Neighbors Classifier to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d array of condition labels
    :param svm_args: additional arguments to pass to :py:class:`~sklearn.neighbors.RadiusNeighborsClassifier`
    :return: 2-tuple of:
        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """
    return sklearn_wrap(
        Z,
        V,
        RadiusNeighborsClassifier(
            **(nc_args or {}),
        ),
    )


@runtime_checkable
class HasPredictBinClass(Protocol):
    """:meta hide-value:"""

    def predict(self, Z: FloatMtx) -> BoolArr: ...


def bin_model(
    Z: FloatMtx,
    model: HasPredictBinClass,
) -> BoolArr:
    """:meta hide-value:"""

    # when possible, check the model is fitted, can't be checked
    # via argument typing since HasEstimator is a Protocol
    if isinstance(model, BaseEstimator):
        check_is_fitted(model)

    return model.predict(Z)


def from_clusts_to_labs(clusts: BoolArr, V: BoolArr, case_only: NumArr) -> BoolArr:
    new_labs = V.copy()

    # cluster 0/1 doesn't necessarily match True/False label so
    # check we check correspondence by using the mean of p_hat in each
    clust_0_has_lower_mean = case_only[~clusts].mean() < case_only[clusts].mean()

    # we only reassign cells in the case condition
    new_labs[V] = clusts if clust_0_has_lower_mean else ~clusts

    return new_labs


def bin_kmeans(
    Y: NumArr,
    V: BoolArr,
    kmeans_args: dict[str, Any] | None = None,
) -> BoolArr:
    """
    runs k-means clustering to binarize continuous scores into boolean labels.

    :param Y: 1-d float array of condition scores (p_hat)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param kmeans_args: additional arguments to pass to :py:class:`~sklearn.linear_model.LogisticRegression` (defaults: ``n_init``: ``"auto"``)
    :return: 1-d boolean array of adjusted condition labels (False corresponds to control, True to case)
    """
    case_only = Y[V]

    return from_clusts_to_labs(
        KMeans(
            n_clusters=2,
            random_state=get_seed(),
            **({"n_init": "auto"} | (kmeans_args or {})),  # pyright: ignore[reportArgumentType]
        )
        .fit_predict(case_only.reshape(-1, 1))
        .astype(bool),
        V,
        case_only,  # pyright: ignore[reportArgumentType]
    )


def bin_gmm(Y: NumArr, V: BoolArr, gmm_args: dict[str, Any] | None = None) -> BoolArr:
    """
    runs k-means clustering to binarize continuous scores into boolean labels.

    :param Y: 1-d float array of condition scores (p_hat)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :return: 1-d boolean array of adjusted condition labels (False corresponds to control, True to case)
    """
    case_only = Y[V]

    return from_clusts_to_labs(
        GaussianMixture(
            n_components=2,
            random_state=get_seed(),
            **(gmm_args or {}),
        )
        .fit_predict(case_only.reshape(-1, 1))
        .astype(bool),
        V,
        case_only,  # pyright: ignore[reportArgumentType]
    )

def bin_argmax_multiclass(
    Y: FloatMtx,
) -> FloatMtx:
    """
    Naive multiclass binarization.
    Returns the index of the highest scoring class for each sample.

    :param Y: n-d float array of probability scores
    :return: 1-d float array of binarized scores
    """

    # return the index of the highest percentaqe class for each sample in Y
    max = np.argmax(Y, axis=1)

    return max

def bin_kmeans_multiclass(
    Y: FloatMtx,
    n_clusters: int | None = None,
    method: Literal["mean", "centroid"] = "centroid",
) -> FloatMtx:
    """
    A k-means based binarization function for multiclass classification.
    Assigns each sample to the label corresponding the highest mean score of k-means clusters in probability space.

    :param Y: n-d float array of probability scores
    :param n_clusters: number of clusters to use for k-means. If None, inferred from the number of columns in Y.
    :param method: method to use for assigning clusters to classes. "centroid" uses the centroid of k-means clusters, "mean" uses the mean of points in the cluster.
    :return: 1-d float array of binarized scores
    """

    if n_clusters is None:
        n_clusters = Y.shape[1]

    kmeans = KMeans(n_clusters=n_clusters, random_state=get_seed())
    cluster_labels = kmeans.fit_predict(Y)

    # Map each cluster to the index of the highest scoring class in that cluster
    cluster_to_class = np.array([])

    for cluster_id in range(n_clusters):
        # indices of samples in this cluster
        idx = np.where(cluster_labels == cluster_id)[0]

        if len(idx) == 0:
            # Handle empty clusters: just map to a dummy class (e.g., argmax of centroid)
            class_id = np.argmax(kmeans.cluster_centers_[cluster_id])
        else:
            if method == "centroid":
                # Get the cluster's centroid (the k-means computed center)
                centroid = kmeans.cluster_centers_[cluster_id]
                # Find the class with the highest probability in the centroid
                class_id = np.argmax(centroid)
            elif method == "mean":
                # Compute mean class-score vector for the cluster
                mean_vec = Y[idx].mean(axis=0)
                # Assign cluster to the class with highest mean score
                class_id = np.argmax(mean_vec)

        cluster_to_class[cluster_id] = class_id
    # print(cluster_to_class)
    # Map every sample's cluster → class
    final_labels = np.take(cluster_to_class, cluster_labels)

    return final_labels


def mannwhitneyu_pvals[T: MatrixLike](lhs: T, rhs: T, lfc_cutoff: float) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    assert lhs.shape[1] == rhs.shape[1], (  # pyright: ignore[reportOptionalSubscript]
        # ignore NECESSITY - spmatrix.shape is not annotated
        "lhs/rhs number of features must be equal"
    )

    ngenes = lhs.shape[1]  # pyright: ignore[reportOptionalSubscript]
    # ignore NECESSITY - spmatrix.shape is not annotated

    if isinstance(lhs, np.ndarray):
        lhs_mean, rhs_mean = np.mean(lhs, axis=0), np.mean(rhs, axis=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
    else:
        lhs_mean, rhs_mean = lhs.mean(axis=0), rhs.mean(axis=0)

    # remove genes for which log2fc is not sensical
    # (i.e. mean of zero in either condition)
    mask = np.logical_and(lhs_mean > 0, rhs_mean > 0)
    # remove genes where log2FC is less than 1.5
    mask[mask] = np.abs(np.log2(rhs_mean[mask] / lhs_mean[mask])) > lfc_cutoff

    lhs, rhs = lhs[:, mask], rhs[:, mask]  # pyright: ignore[reportAssignmentType]
    # ignore NECESSITY - pyright can't tell that lhs/rhs are of type MatrixLike
    # due to type hints on indexing not sufficiently preserving type info

    # mannwhitneyu does not work on sparse arrays
    if not isinstance(lhs, np.ndarray):
        lhs, rhs = lhs.todense(), rhs.todense()  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
        # ignore NECESSITY - pyright can't tell that lhs/rhs are of type sparray

    pvals = np.full((ngenes,), np.nan)
    pvals[mask] = mannwhitneyu(lhs, rhs, axis=0).pvalue * ngenes
    return pvals


def mannwhitneyu_ndeg[T: MatrixLike](lhs: T, rhs: T, lfc_cutoff: float, signif_cutoff: float = 0.05) -> Integral:
    return np.sum((mannwhitneyu_pvals(lhs, rhs, lfc_cutoff)) < signif_cutoff)  # pyright: ignore[reportReturnType]


def score_deg(
    X: MatrixLike,
    W: BoolArr,
    deg_cutoff: float = 1.5,
) -> Integral:
    """
    implements a heuristic to score label adjustment based on number of DEGs produced by new labels.

    :param X: input cell by gene matrix
    :param V: 1-d boolean array of condition labels (True corresponds to case, False to control)
    :param W: 1-d boolean array of adjusted condition labels (True corresponds to case, False to control)
    :return: number of DEGs between conditions (as determined by a Bonferroni-corrected Mann-Whitney U test p value of less than 0.05 and an absolute log2 fold change of more than 1.5)
    """

    lhs, rhs = X[~W, :], X[W, :]
    return mannwhitneyu_ndeg(
        lhs,  # pyright: ignore[reportArgumentType]
        rhs,  # pyright: ignore[reportArgumentType]
        # ignore NECESSITY - pyright can't tell that lhs/rhs are of type MatrixLike
        # due to type hints on indexing not sufficiently preserving type info
        deg_cutoff,
    )


def score_deg2(
    X: MatrixLike,
    V: BoolArr,
    W: BoolArr,
    score_weight_relab: float = 1.0,
    score_weight_vsctl: float = 1.0,
    deg_cutoff: float = 1.5,
) -> Integral:
    """
    implements a heuristic to score label adjustment based on number of DEGs produced by new labels using two comparisons.
    specifically two comparisons are run, and the return value is a weighted difference of:

    1) case cells labeled affected vs case cells labeled unaffected (difference should be as large as possible)
    2) control cells vs case cells labeled unaffected (difference should be as small as possible)

    :param X: input cell by gene matrix
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param W: boolean array of adjusted condition labels (False corresponds to control, True to case)
    :param score_weight_relab: weight given to number of DEGs in first comparison
    :param score_weight_vsctl: weight given to number of DEGs in second comparison
    :return: ``score_weight_relab`` * number of DEGs in first comparison - ``score_weight_vsctl`` * number of DEGs in second comparison
    """
    case_only_expr = X[V, :]
    case_only_vhat = W[V]
    ctl_only_expr = X[~V, :]

    return (
        mannwhitneyu_ndeg(
            # X values from labeled unaffected
            case_only_expr[~case_only_vhat],  # pyright: ignore[reportOperatorIssue, reportArgumentType]
            # X values from labeled affected
            case_only_expr[case_only_vhat],  # pyright: ignore[reportArgumentType]
            deg_cutoff,
        )
        * score_weight_relab
    ) - (
        mannwhitneyu_ndeg(
            # X values from true control
            ctl_only_expr,  # pyright: ignore[reportArgumentType]
            # X values from labeled unaffected
            case_only_expr[~case_only_vhat],  # pyright: ignore[reportArgumentType]
            deg_cutoff,
        )
        * score_weight_vsctl
    )
    # ignore NECESSITY - ndarray indexing type bounds not specific
    # enough to show that they confirm to MatrixLike shape


def score_phatdiff(
    Y: NumArr,
    V: BoolArr,
    W: BoolArr,
    score_weight_relab: float = 1.0,
    score_weight_vsctl: float = 1.0,
) -> NumericScalar:
    """
    implements a heuristic to score label adjustment based on the distribution difference (using the Kolmogorov-Smirnov statistic) between p_hat values.
    specifically two comparisons are run, and the return value is a weighted difference of:

    1) case cells labeled affected vs case cells labeled unaffected (difference should be as large as possible)
    2) control cells vs case cells labeled unaffected (difference should be as small as possible)

    :param Y: 1-d float array of condition scores (p_hat)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param W: boolean array of adjusted condition labels (False corresponds to control, True to case)
    :param score_weight_relab: weight given to KS stat of the first comparison
    :param score_weight_vsctl: weight given to KS stat of the second comparison
    :return: ``score_weight_relab`` * KS stat of first comparison - ``score_weight_vsctl`` * KS stat of second comparison
    """
    case_only_pred = Y[V]
    case_only_vhat = W[V]
    ctl_only_pred = Y[~V]

    return (
        ks_2samp(
            # Y values from labeled unaffected
            case_only_pred[~case_only_vhat],
            # Y values from labeled affected
            case_only_pred[case_only_vhat],
            alternative="greater",
            # `ks_2samp` alternative refers to the CDFs of `data1`/`data2`
            # the trend in which is inversely related to the trend of
            # the means of `data1`/`data2`. as such, since our alternative
            # hypothesis is that labeled unaffected cells have a p_hat
            # distribution less than that of labeled affected cells,
            # the alternative argument is set to "greater", as
            # the trend for their respective CDFs would be inverted
        ).statistic  # pyright: ignore[reportAttributeAccessIssue]
        * score_weight_relab
    ) - (
        ks_2samp(
            # Y values from true control
            ctl_only_pred,
            # Y values from labeled unaffected
            case_only_pred[~case_only_vhat],
        ).statistic  # pyright: ignore[reportAttributeAccessIssue]
        * score_weight_vsctl
    )
    # ignore NECESSITY - `scipy.stats.ks_2samp` lacking proper
    # annotation on return type (currently set to _)


def symm_kl_div(lhs: NumArr, rhs: NumArr) -> NumericScalar:
    """
    implements a symmetrized version of Kullback-Leibler (KL) divergence, by returning the sum of the KL divergence from ``lhs`` to ``rhs`` and from ``rhs`` to ``lhs``.
    this symmetrized statistic is also known as the Jeffreys divergence.

    :param lhs: 1-d N-length vector of probability values
    :param rhs: 1-d N-length vector of probability values
    :returns: KL divergence of ``lhs`` to ``rhs`` + KL divergence of ``rhs`` to ``lhs``
    """
    return np.sum(kl_div(lhs, rhs)) + np.sum(kl_div(rhs, lhs))


def score_phatdiff_dist(
    Y: NumArr,
    V: BoolArr,
    W: BoolArr,
    distance_fn: Callable[[NumArr, NumArr], NumericScalar] = wasserstein_distance,
    score_weight_relab: float = 1.0,
    score_weight_vsctl: float = 1.0,
) -> NumericScalar:
    """
    implements a heuristic to score label adjustment based on the distribution difference between p_hat values.
    specifically two comparisons are run, and the return value is a weighted difference of:

    1) case cells labeled affected vs case cells labeled unaffected (difference should be as large as possible)
    2) control cells vs case cells labeled unaffected (difference should be as small as possible)

    :param Y: 1-d float array of condition scores (p_hat)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param W: boolean array of adjusted condition labels (False corresponds to control, True to case)
    :param distance_fn: distance function used to calculate distribution difference between p_hat values (defaults to earth's mover distance, as computed by :py:func:`~scipy.stats.wasserstein_distance`)
    :param score_weight_relab: weight given to difference of the first comparison
    :param score_weight_vsctl: weight given to difference of the second comparison
    :return: ``score_weight_relab`` * statistic from first comparison - ``score_weight_vsctl`` * statistic from second comparison
    """
    case_only_pred = Y[V]
    case_only_vhat = W[V]
    ctl_only_pred = Y[~V]

    return (
        distance_fn(
            # Y values from labeled unaffected
            case_only_pred[~case_only_vhat],  # pyright: ignore[reportArgumentType]
            # Y values from labeled affected
            case_only_pred[case_only_vhat],  # pyright: ignore[reportArgumentType]
        )
        * score_weight_relab
    ) - (
        distance_fn(
            # Y values from true control
            ctl_only_pred,  # pyright: ignore[reportArgumentType]
            # Y values from labeled unaffected
            case_only_pred[~case_only_vhat],  # pyright: ignore[reportArgumentType]
        )
        * score_weight_vsctl
    )
    # ignore NECESSITY - ndarray indexing type bounds not specific
    # enough to show that they confirm to NumArr shape


def score_nulldist(
    Y: NumArr,
    V: BoolArr,
    pipeline_algo: Pipeline,
    n_iters: int = 10,
    distance_fn: Callable[[NumArr, NumArr], NumericScalar] = wasserstein_distance,
    **kwargs,
) -> NumericScalar:
    """
    implements a heuristic to score label adjustment based on a distance metric between p_hat values generated by the pipeline and
    an approximation of null-distributed p_hat values (generated by re-running the pipeline regression step given randomly permuted condition labels).

    :param Y: 1-d float array of condition scores (p_hat)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param pipeline_algo: pipeline class used to generate outputs
    :param n_iters: number of iterations to run random sampling procedure over, returning average of all iterations
    :param distance_fn: distance metric to use between "null" Y values and pipeline outputs (defaults to earth's mover distance, as computed by :py:func:`~scipy.stats.wasserstein_distance`)
    :return: earth mover's distance between pipeline generated p_hat values and p_hat values yielded by re-running regression on randomly permuted condition labels
    """

    rng = np.random.default_rng(get_seed())
    return (
        sum(
            [
                distance_fn(
                    wcall(
                        kwargs | {"V": rng.permuted(V)},
                        out_to_dict(pipeline_algo.regr_fn, getattr(pipeline_algo.regr_fn, _INTERNAL_WRAP_ATTR_NAME)),
                        pipeline_algo.strict,
                    )["Y"],  # pyright: ignore[reportArgumentType]
                    # ignore NECESSITY - Y should always be a NumArr
                    Y,
                )
                for _ in range(n_iters)
            ]
        )
        / n_iters
    )
