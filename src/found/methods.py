from collections.abc import Callable
from numbers import Integral
from typing import Any, Protocol, Self, runtime_checkable
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
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import sparsefuncs

from .adapters import _INTERNAL_WRAP_ATTR_NAME, Pipeline, out_to_dict, step_fn, wcall
from .seed import get_seed
from .types import BoolArr, FloatMtx, MatrixLike, NumArr, NumericScalar


def mult_preserve_type[T: MatrixLike](lhs: T, rhs: np.ndarray) -> T:
    o = lhs * rhs

    if isinstance(lhs, sp.csr_array):
        o = o.tocsr()
    elif isinstance(lhs, sp.csc_array):
        o = o.tocsc()

    assert isinstance(o, type(lhs))
    return o


def log1p[T: MatrixLike](X: T) -> T:
    """
    matrix format aware log1p

    :param X: input matrix
    :return: ``log(X + 1)``
    """

    if isinstance(X, sp.csr_array):
        X = X.log1p()  # ty:ignore[unresolved-attribute]
        assert isinstance(X, sp.csr_array)
    elif isinstance(X, sp.csc_array):
        X = X.log1p()  # ty:ignore[unresolved-attribute]
        assert isinstance(X, sp.csc_array)
    # ignore NECESSITY - typchecker can't detect log1p method on sparse matrix types
    # because method is added dynamically, as observed here:
    # https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/_data.py#L138
    # https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/_base.py#L52
    else:
        X = np.log1p(X)  # ty:ignore[no-matching-overload]
        # ignore NECESSITY - typechecker doesn't narrow X to an ndarray

    return X


def scale_rs[T: MatrixLike](X: T) -> T:
    """
    returns scaled version of input matrix with row-wise sums of 1

    :param X: input matrix
    :return: scaled matrix
    """
    per_cell_sum = X.sum(axis=1)  # ty:ignore[invalid-argument-type]
    # ignore NECESSITY - ???
    avg_counts_per_cell = per_cell_sum.mean()
    size_fact = per_cell_sum / avg_counts_per_cell

    # silence warnings about reciprocal for zero, filled with zeros elsewhere
    with catch_warnings(action="ignore", category=RuntimeWarning):
        recip = np.reciprocal(size_fact)
    recip = np.where(size_fact > 0, recip, 0.0)

    return mult_preserve_type(X, recip[:, np.newaxis])


def scale_sd[T: MatrixLike](X: T) -> T:
    """
    returns scaled version of input matrix with column-wise standard deviation of 1

    :param X: input matrix
    :return: scaled matrix
    """

    if isinstance(X, np.ndarray):
        stdev = np.std(X, axis=0)  # ty:ignore[no-matching-overload]
        # ignore NECESSITY - ???
    else:
        _, var = sparsefuncs.mean_variance_axis(X, axis=0)
        # note: sparsefuncs.mean_variance_axis return type set to Unknown, however type is documented to be 2-tuple given these arguments
        stdev = np.sqrt(var)

    # silence warnings about reciprocal for zero, filled with zeros elsewhere
    with catch_warnings(action="ignore", category=RuntimeWarning):
        recip = np.reciprocal(stdev)
    recip = np.where(stdev > 0, recip, 0.0)

    return mult_preserve_type(X, recip[np.newaxis, :])


def vst_shiftlog[T: MatrixLike](X: T, overdispersion: float = 0.05) -> T:
    """
    implements a variance stabilizing transform accounting for overdispersion, with size factors scaled to have a geometric mean of 1,
    as recommended in `Ahlmann-Eltze et al. <https://doi.org/10.1038/s41592-023-01814-1>`_.

    :param X: input matrix
    :param overdispersion: overdispersion factor
    :return: variance stabilized matrix
    """
    per_cell_sum = X.sum(axis=1)  # ty:ignore[invalid-argument-type]
    # ignore NECESSITY - ???
    size_fact = per_cell_sum / np.exp(np.mean(np.log(per_cell_sum)))

    # silence warnings about reciprocal for zero, filled with zeros elsewhere
    with catch_warnings(action="ignore", category=RuntimeWarning):
        recip = np.reciprocal(size_fact)
    recip = np.where(size_fact > 0, recip, 0.0) * 4 * overdispersion

    X = mult_preserve_type(X, recip[:, np.newaxis])

    X = log1p(X)

    X = X * np.reciprocal(np.sqrt(overdispersion))

    return X


def run_pca(
    X: MatrixLike | sp.csc_matrix | sp.csr_matrix,
    k: int,
    pre_pca_tf: Callable[[MatrixLike], MatrixLike] | None = lambda X: scale_sd(vst_shiftlog(X)),
    pca_args: dict[str, Any] | None = None,
) -> FloatMtx:
    """
    runs PCA as implemented in :py:class:`~sklearn.decomposition.PCA` using the ARPACK solver to provide PC-space embeddings for the input matrix.
    an optional callback can be provided to transform data prior to PCA (e.g. variance stabilizing transform, scaling, etc.).

    note: :py:class:`~sklearn.decomposition.PCA` w/ ARPACK handles center-ing internally, so the provided transform **should not** center provided data.

    :param X: cell by gene matrix
    :param k: dimensionality of the PCA to run
    :param pre_pca_tf: optional callback to transform data matrix prior to PCA embedding, defaults to :py:func:`~found.methods.vst_shiftlog` followed by :py:func:`~found.methods.scale_sd`
    :param pca_args: additional arguments to pass to :py:class:`~sklearn.decomposition.PCA`
    :return: cell by k matrix representing cells in PC-space
    """
    # if dealing with spmatrix types, convert to sparray equivalents
    if isinstance(X, sp.csr_matrix):
        X = sp.csr_array(X)
    if isinstance(X, sp.csc_matrix):
        X = sp.csc_array(X)

    if pre_pca_tf is not None:
        X = pre_pca_tf(X)

    return PCA(
        k,
        random_state=get_seed(),
        svd_solver="arpack",
        **(pca_args or {}),
    ).fit_transform(X)


@step_fn("Z", "NMF_p")
def run_nmf(
    X: MatrixLike | sp.csc_matrix | sp.csr_matrix,
    k: int,
    pre_nmf_tf: Callable[[MatrixLike], MatrixLike] | None = lambda X: scale_sd(scale_rs(X)),
    nmf_args: dict[str, Any] | None = None,
) -> tuple[FloatMtx, FloatMtx]:
    """
    runs NMF (decomposition of ``X`` into ``w @ h``) as implemented in :py:class:`~sklearn.decomposition.NMF` to provide NMF-cell-by-k-space coordinates for each cell.

    note:

    - ``w`` refers to the cell by k matrix
    - ``h`` refers to the k by gene matrix

    see :py:class:`~sklearn.decomposition.NMF` documentation for more details on meaning of arguments.

    :param X: cell by gene matrix
    :param k: k, specifying number of NMF components/programs
    :param pre_nmf_tf: optional callback to transform data matrix prior to NMF embedding, defaults to :py:func:`~found.methods.scale_rs` followed by :py:func:`~found.methods.scale_sd`
    :param nmf_args: additional arguments to pass to :py:class:`~sklearn.decomposition.NMF`
    :return: 2-tuple of:

        - cell by k matrix representing cells in NMF-cell-by-k-space (e.g. ``w``)
        - gene by k matrix representing NMF-computed gene programs (e.g. ``h^T``)
    """
    # if dealing with spmatrix types, convert to sparray equivalents
    if isinstance(X, sp.csr_matrix):
        X = sp.csr_array(X)
    if isinstance(X, sp.csc_matrix):
        X = sp.csc_array(X)

    if pre_nmf_tf is not None:
        X = pre_nmf_tf(X)

    m = NMF(k, random_state=get_seed(), **(nmf_args or {}))
    z = m.fit_transform(X)

    return z, m.components_.T


class HasFitBinClass(Protocol):
    classes_: np.ndarray[tuple[int], np.dtype[np.bool]]

    fit: Callable[[FloatMtx, BoolArr], Self]


def sklearn_wrap[T: HasFitBinClass, U: FloatMtx](
    Z: U,
    V: BoolArr,
    model: T,
    score_fn: Callable[
        [T, U],
        np.ndarray[tuple[int], np.dtype[np.floating]],
    ] = lambda m, z: m.predict_proba(z)[
        # ignore NECESSITY - predict_proba method not present on HasFitBinClass
        # on purpose to allow for compatibility with SVM use case, but is _generally_ present
        # so it is useful to keep this as default argument value to avoid repetition of below snippet
        :, m.classes_.tolist().index(True)
        # classes_ is generated by np.unique, which returns the classes
        # in sorted order, so True should always be the second class
        # however we still perform the `.index` operation as a sanity check
    ],
) -> tuple[NumArr, T]:
    model = model.fit(Z, V)

    return (score_fn(model, Z), model)


@step_fn("Y", "model")
def reg_logit(
    Z: FloatMtx,
    V: BoolArr,
    logit_args: dict[str, Any] | None = None,
) -> tuple[NumArr, LogisticRegression]:
    """
    runs a logistic regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param logit_args: additional arguments to pass to :py:class:`~sklearn.linear_model.LogisticRegression` (defaults: ``solver``: ``"newton-cg"``, ``C``: :py:obj:`~numpy.inf`)
    :return: 2-tuple of:

        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """
    with catch_warnings(action="ignore", category=UserWarning, lineno=1170):
        return sklearn_wrap(
            Z,
            V,
            LogisticRegression(
                random_state=get_seed(),
                **({"solver": "newton-cg", "C": np.inf} | (logit_args or {})),
            ),
        )


@step_fn("Y", "model")
def reg_svm(
    Z: FloatMtx,
    V: BoolArr,
    svm_args: dict[str, Any] | None = None,
) -> tuple[NumArr, SVC]:
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
            random_state=get_seed(),
            **(svm_args or {}),
        ),
        lambda m, z: m.decision_function(z),
    )


@step_fn("Y", "model")
def reg_lsvm(
    Z: FloatMtx,
    V: BoolArr,
    lsvm_args: dict[str, Any] | None = None,
) -> tuple[NumArr, LinearSVC]:
    """
    runs a linear kernel support vector machine classifier -based regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param lsvm_args: additional arguments to pass to :py:class:`~sklearn.svm.LinearSVC`
    :return: 2-tuple of:

        - 1-d float array of probability scores generated by the fitted model
        - the fitted model object
    """
    return sklearn_wrap(
        Z,
        V,
        LinearSVC(
            random_state=get_seed(),
            **(lsvm_args or {}),
        ),
        lambda m, z: m.decision_function(z),
    )


@step_fn("Y", "model")
def reg_gp(
    Z: FloatMtx,
    V: BoolArr,
    gp_args: dict[str, Any] | None = None,
) -> tuple[NumArr, GaussianProcessClassifier]:
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
) -> tuple[NumArr, RandomForestClassifier]:
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
        RandomForestClassifier(  # ty:ignore[invalid-argument-type]
            # ignore NECESSITY - `RandomForestClassifier.classes_` attribute is set to be list[Unknown], not just ndarray
            random_state=get_seed(),
            **(rf_args or {}),
        ),
    )


@runtime_checkable
class HasFitPredictClass(Protocol):
    """:meta hide-value:"""

    def fit_predict(
        self, X: np.ndarray[tuple[int, int], np.dtype[np.number]]
    ) -> np.ndarray[tuple[int], np.dtype[np.integer]]: ...


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


def from_clusts_to_labs(model: HasFitPredictClass, V: BoolArr, case_only_Y_hat: NumArr) -> BoolArr:
    clusts = model.fit_predict(case_only_Y_hat[:, np.newaxis]).astype(bool)
    new_labs = V.copy()

    # cluster 0/1 doesn't necessarily match True/False label so
    # check we check correspondence by using the mean of p_hat in each
    if len(cl := np.unique(clusts)) == 1:
        clust_0_has_lower_mean = bool(cl[0]) == (case_only_Y_hat[clusts == cl[0]].mean() > 0.5)
    else:
        clust_0_has_lower_mean = case_only_Y_hat[~clusts].mean() < case_only_Y_hat[clusts].mean()

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
    :param kmeans_args: additional arguments to pass to :py:class:`~sklearn.cluster.KMeans` (defaults: ``n_init``: ``"auto"``)
    :return: 1-d boolean array of adjusted condition labels (False corresponds to control, True to case)
    """

    return from_clusts_to_labs(
        KMeans(
            n_clusters=2,
            random_state=get_seed(),
            **({"n_init": "auto"} | (kmeans_args or {})),
        ),
        V,
        Y[V],
    )


def bin_gmm(Y: NumArr, V: BoolArr, gmm_args: dict[str, Any] | None = None) -> BoolArr:
    """
    runs k-means clustering to binarize continuous scores into boolean labels.

    :param Y: 1-d float array of condition scores (p_hat)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param gmm_args: additional arguments to pass to :py:class:`~sklearn.mixture.GaussianMixture`
    :return: 1-d boolean array of adjusted condition labels (False corresponds to control, True to case)
    """

    return from_clusts_to_labs(
        GaussianMixture(
            n_components=2,
            random_state=get_seed(),
            **(gmm_args or {}),
        ),
        V,
        Y[V],
    )


def mannwhitneyu_pvals[T: MatrixLike](lhs: T, rhs: T, lfc_cutoff: float) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    assert lhs.shape[1] == rhs.shape[1], "lhs/rhs number of features must be equal"

    ngenes = lhs.shape[1]

    if isinstance(lhs, np.ndarray):
        lhs_mean, rhs_mean = np.mean(lhs, axis=0), np.mean(rhs, axis=0)  # ty:ignore[no-matching-overload]
        # ignore NECESSITY - ???
    else:
        lhs_mean, rhs_mean = lhs.mean(axis=0), rhs.mean(axis=0)  # ty:ignore[invalid-argument-type]
        # ignore NECESSITY - ???

    # remove genes for which log2fc is not sensical
    # (i.e. mean of zero in either condition)
    mask = np.logical_and(lhs_mean > 0, rhs_mean > 0)
    # remove genes where log2FC is less than 1.5
    mask[mask] = np.abs(np.log2(rhs_mean[mask] / lhs_mean[mask])) > lfc_cutoff

    lhs, rhs = lhs[:, mask], rhs[:, mask]  # ty:ignore[invalid-argument-type]
    # ignore NECESSITY - typechecker can't tell that lhs/rhs are of type MatrixLike
    # due to type hints on indexing not sufficiently preserving type info

    # mannwhitneyu does not work on sparse arrays
    if not isinstance(lhs, np.ndarray):
        lhs, rhs = lhs.todense(), rhs.todense()

    pvals = np.full((ngenes,), np.nan)
    pvals[mask] = mannwhitneyu(lhs, rhs, axis=0).pvalue * ngenes
    return pvals


def mannwhitneyu_ndeg[T: MatrixLike](lhs: T, rhs: T, lfc_cutoff: float, signif_cutoff: float = 0.05) -> np.integer:
    # mannwhitneyu_pvals returns nans for failed comparisons, but np.nan < x = False for all x so this works
    return np.sum((mannwhitneyu_pvals(lhs, rhs, lfc_cutoff)) < signif_cutoff)  # ty:ignore[invalid-return-type]
    # ignore NECESSITY - ???


def score_deg(X: MatrixLike, W: BoolArr, lfc_cutoff: float = 1.5, sig_cutoff: float = 0.05) -> np.integer:
    """
    implements a heuristic to score label adjustment based on number of DEGs produced by new labels.

    :param X: input cell by gene matrix
    :param W: 1-d boolean array of adjusted condition labels (True corresponds to case, False to control)
    :param lfc_cutoff: threshold log2 fold change value for a gene to be considered a DEG
    :param sig_cutoff: threshold Bonferroni-corrected Mann-Whitney U-test p value for a gene to be considered a DEG
    :return: number of DEGs (as determined by a Bonferroni-corrected Mann-Whitney U-test p value of less than ``sig_cutoff`` and an absolute log2 fold change of more than ``lfc_cutoff``)
    """

    lhs, rhs = X[~W, :], X[W, :]
    return mannwhitneyu_ndeg(lhs, rhs, lfc_cutoff, sig_cutoff)


def score_deg2(
    X: MatrixLike,
    V: BoolArr,
    W: BoolArr,
    score_weight_vsctl: float = 1.0,
    lfc_cutoff: float = 1.5,
    sig_cutoff: float = 0.05,
) -> Integral:
    """
    implements a heuristic to score label adjustment based on number of DEGs produced by new labels using two comparisons.
    specifically two comparisons are run, and the return value is a weighted difference of:

    1) case cells labeled affected vs case cells labeled unaffected (amount should be as large as possible)
    2) control cells vs case cells labeled unaffected (amount should be as small as possible)

    :param X: input cell by gene matrix
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param W: boolean array of adjusted condition labels (False corresponds to control, True to case)
    :param score_weight_vsctl: weight given to number of DEGs in second comparison
    :param lfc_cutoff: threshold log2 fold change value for a gene to be considered a DEG
    :param sig_cutoff: threshold Bonferroni-corrected Mann-Whitney U-test p value for a gene to be considered a DEG
    :return: number of DEGs in first comparison - ``score_weight_vsctl`` * number of DEGs in second comparison
    """
    case_only_expr = X[V, :]
    case_only_vhat = W[V]
    ctl_only_expr = X[~V, :]

    return mannwhitneyu_ndeg(
        # X values from labeled unaffected
        case_only_expr[~case_only_vhat],
        # X values from labeled affected
        case_only_expr[case_only_vhat],
        lfc_cutoff,
        sig_cutoff,
    ) - (
        mannwhitneyu_ndeg(
            # X values from true control
            ctl_only_expr,
            # X values from labeled unaffected
            case_only_expr[~case_only_vhat],
            lfc_cutoff,
            sig_cutoff,
        )
        * score_weight_vsctl
    )


def score_ks_diff(
    Y: NumArr,
    V: BoolArr,
    W: BoolArr,
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
    :param score_weight_vsctl: weight given to KS stat of the second comparison
    :return: KS stat of first comparison - ``score_weight_vsctl`` * KS stat of second comparison
    """
    case_only_pred = Y[V]
    case_only_vhat = W[V]
    ctl_only_pred = Y[~V]

    return ks_2samp(
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
    ).statistic - (
        ks_2samp(
            # Y values from true control
            ctl_only_pred,
            # Y values from labeled unaffected
            case_only_pred[~case_only_vhat],
        ).statistic
        * score_weight_vsctl
    )


def symm_kl_div(lhs: NumArr, rhs: NumArr) -> NumericScalar:
    """
    implements a symmetrized version of Kullback-Leibler (KL) divergence, by returning the sum of the KL divergence from ``lhs`` to ``rhs`` and from ``rhs`` to ``lhs``.
    this symmetrized statistic is also known as the Jeffreys divergence.

    :param lhs: 1-d N-length vector of probability values
    :param rhs: 1-d N-length vector of probability values
    :returns: KL divergence of ``lhs`` to ``rhs`` + KL divergence of ``rhs`` to ``lhs``
    """
    return np.sum(kl_div(lhs, rhs)) + np.sum(kl_div(rhs, lhs))


def score_dist_diff(
    Y: NumArr,
    V: BoolArr,
    W: BoolArr,
    distance_fn: Callable[[NumArr, NumArr], NumericScalar] = wasserstein_distance,
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
    :param score_weight_vsctl: weight given to result of second comparison
    :return: statistic from first comparison - ``score_weight_vsctl`` * statistic from second comparison
    """
    case_only_vhat = W[V]
    if len(np.unique(case_only_vhat)) == 1:
        return 0  # @TODO: figure out better strategy to deal with no/total relabeling

    case_only_pred = Y[V]
    ctl_only_pred = Y[~V]

    return distance_fn(
        # Y values from labeled unaffected
        case_only_pred[~case_only_vhat],
        # Y values from labeled affected
        case_only_pred[case_only_vhat],
    ) - (
        distance_fn(
            # Y values from true control
            ctl_only_pred,
            # Y values from labeled unaffected
            case_only_pred[~case_only_vhat],
        )
        * score_weight_vsctl
    )


def score_null_dist(
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
    :return: distance between pipeline generated p_hat values and p_hat values yielded by re-running regression on randomly permuted condition labels
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
                    )["Y"],  # ty:ignore[invalid-argument-type]
                    # ignore NECESSITY - Y should always be a NumArr
                    Y,
                )
                for _ in range(n_iters)
            ]
        )
        / n_iters
    )
