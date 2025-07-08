import warnings
from numbers import Real

import numpy as np
import scipy.sparse as sp
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.utils import sparsefuncs

from .types import BoolArr, FloatMtx, MatrixLike, NumArr

# TODO: should this be behind a multiprocessing.Lock (?)
_RAND_SEED = None


def set_seed(seed: int | None):
    """
    conveninence function used to set a singular random seed for all methods utilized in this module, with the aim of guaranteeing reproducibility.

    :param seed: integer seed, must be within [0, 4294967295], set to None to remove fixed seeding
    """
    global _RAND_SEED
    if seed is not None:
        if seed < 0 or seed > 4294967295:
            raise ValueError(f"provided seed {seed} outside of allowed range [0, 4294967295]")
    _RAND_SEED = seed


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

    per_cell_sum = X.sum(axis=1)
    avg_counts_per_cell = per_cell_sum.mean()
    size_fact = per_cell_sum / avg_counts_per_cell
    X = X / size_fact[:, np.newaxis]

    if isinstance(X, np.ndarray):
        X = np.log1p(X)  # pyright: ignore
        # ignore NECESSITY - log1p method type hint does not preserve shape type hint
    else:
        X = X.tocsr().log1p()  # pyright: ignore
        # ignore NECESSITY - pyright can't detect log1p method on sparse matrix types
        # because method is addded dynamically, as observed here:
        # https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/_data.py#L138
        # https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/_base.py#L52

    # make sure we're not introducing sp.spmatrix type further into the pipeline
    assert not isinstance(X, sp.spmatrix)

    return X


def per_gene_scale(N: MatrixLike, center: bool = True, scale: bool = True) -> FloatMtx:
    if not (center or scale):
        if not isinstance(N, np.ndarray):
            return N.todense()  # pyright: ignore
        return N.astype(float)

    if isinstance(N, np.ndarray):
        mean = np.mean(N, axis=0)
        stdev = np.std(N, axis=0)
    else:
        mean, var = sparsefuncs.mean_variance_axis(N, axis=0)  # pyright: ignore
        # ignore NECESSITY - sparsefuncs.mean_variance_axis return type set to
        # Unknown, however type is documented to be 2-tuple given these arguments
        stdev = np.sqrt(var)

    scaled = N

    if center:
        scaled = N - mean
    if scale:
        # silence division by zero error, then fill NaN with 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = scaled / stdev
        scaled = np.nan_to_num(scaled, nan=0.0)

    return scaled


def run_pca(N: MatrixLike, k: int) -> FloatMtx:
    """
    runs PCA to provide PC-space coordinates for each cell.

    :param N: cell by gene matrix
    :param k: dimensionality of the PCA to run
    :return: cell by k matrix representing cells in PC-space
    """

    # return PCA output
    return PCA(k, svd_solver="arpack", random_state=_RAND_SEED).fit_transform(per_gene_scale(N))


def log_reg(Z: FloatMtx, V: BoolArr) -> NumArr:
    """
    runs a logistic regression to score cells as affected/unaffected by the condition.

    :param Z: cell by k matrix (where k is some number of dimensions)
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :return: 1-d float array of probability scores from the fitted logistic regression model
    """
    return (
        LogisticRegression(
            penalty=None,  # pyright: ignore
            # ignore NECESSITY - penalty can be None but type set to str
            random_state=_RAND_SEED,
        )
        .fit(Z, V)
        .predict_proba(Z)[:, 1]
    )


def kmeans_bin(Y: NumArr, V: BoolArr) -> BoolArr:
    """
    runs k-means clustering to binarize continuous scores into boolean labels.

    :param Y: 1-d float array of condition scores
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :return: 1-d boolean array of adjusted condition labels (False corresponds to control, True to case)
    """
    case_only = Y[V]
    clusts: BoolArr = (
        KMeans(n_clusters=2, n_init="auto", random_state=_RAND_SEED).fit_predict(case_only.reshape(-1, 1)).astype(bool)
    )

    new_labs = V.copy()

    # cluster 0/1 doesn't necessarily match True/False label so
    # check we check correspondence by using the mean of p_hat in each
    clust_0_has_lower_mean = case_only[~clusts].mean() < case_only[clusts].mean()
    new_labs[V] = clusts if clust_0_has_lower_mean else ~clusts

    return new_labs


def score_deg(X: MatrixLike, V: BoolArr, W: BoolArr) -> Real:
    """
    implements a heuristic to score label adjustment based on number of DEGs produced by new labels.

    :param X: input cell by gene matrix
    :param V: 1-d boolean array of condition labels (True correspons to case, False to control)
    :param W: 1-d boolean array of adjusted condition labels (True correspons to case, False to control)
    :return: number of DEGs between conditions (as determined by a bonferroni-corrected mann-whitney U test p value of less than 0.05 and an absolute log2 fold change of more than 1.5)
    """
    unaffected = X[W, :]
    affected = X[~W, :]

    ngenes = X.shape[1]  # pyright: ignore
    # ignore NECESSITY - spmatrix.shape is not annotated
    # to be a tuple, inferred annotations set it to None

    abs_lfc = np.log2(np.mean(affected, axis=0) / np.mean(unaffected, axis=1)).abs()

    # TODO: find way to vectorize (if even possible ?)
    # for loop over dimension of matrix is a bad pattern
    return sum(
        (mannwhitneyu(unaffected[:, col], affected[:, col]).pvalue * ngenes < 0.05) and abs_lfc[col] > 1.5
        for col in range(ngenes)
    )  # pyright: ignore
    # ignore NECESSITY - int not assignable to Real


def score_ks(Y: NumArr, V: BoolArr, W: BoolArr) -> Real:
    """
    implements a heuristic to score label adjustment based on the Kolmogorov-Smirnov test statistic between p_hat values in the case condition.

    :param Y: 1-d float array of condition scores
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param W: boolean array of adjusted condition labels (False corresponds to control, True to case)
    :return: 2 sample Kolmogorov-Smirnov test statistic value between ``Y`` values for unaffected vs affected in the originally labelled case condition.
    """
    case_only_pred = Y[V]
    case_only_vhat = W[V]
    return ks_2samp(
        case_only_pred[~case_only_vhat],  # Y values from labelled unaffected
        case_only_pred[case_only_vhat],  #  Y values from labelled affected
        alternative="greater",
    ).statistic  # pyright:ignore
    # ignore NECESSITY - scipy.stats.ks_2samp lacking proper
    # annotation on return type (currently set to _)
