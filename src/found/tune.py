from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

from .adapters import _INTERNAL_WRAP_ATTR_NAME, Pipeline, out_to_dict, wcall
from .types import BoolArr, FloatMtx, MatrixLike, NumArr


class Tuner[HyperparameterType, ScoreType](ABC):
    """
    abstract base class for a hyper parameter tuner which wraps some pipeline
    """

    @abstractmethod
    def __call__(
        self, algo: Pipeline, **kwargs
    ) -> tuple[HyperparameterType, Mapping[HyperparameterType, tuple[NumArr, BoolArr, ScoreType]]]: ...


def mannwhitneyu_ndeg[T: MatrixLike](lhs: T, rhs: T, deg_cutoff: float) -> Integral:
    assert lhs.shape[1] == rhs.shape[1], (  # pyright: ignore
        # ignore NECESSITY - spmatrix.shape is not annotated
        "lhs/rhs number of features must be equal"
    )

    ngenes = lhs.shape[1]  # pyright: ignore
    # ignore NECESSITY - spmatrix.shape is not annotated

    if isinstance(lhs, np.ndarray):
        lhs_mean, rhs_mean = np.mean(lhs, axis=0), np.mean(rhs, axis=0)  # pyright: ignore

    else:
        lhs_mean, rhs_mean = lhs.mean(axis=0), rhs.mean(axis=0)

    # remove genes for which log2fc is not sensical
    # (i.e. mean of zero in either condition)
    gtz = np.logical_and(lhs_mean > 0, rhs_mean > 0)
    # remove genes where log2FC is less than 1.5
    lfc = np.abs(np.log2(rhs_mean[gtz] / lhs_mean[gtz])) > deg_cutoff

    lhs, rhs = lhs[:, gtz][:, lfc], rhs[:, gtz][:, lfc]  # pyright: ignore
    # ignore NECESSITY - pyright can't tell that lhs/rhs are of type MatrixLike
    # due to type hints on indexing not sufficiently preserving type info

    # mannwhitneyu does not work on sparse arrays
    if not isinstance(lhs, np.ndarray):
        lhs, rhs = lhs.todense(), rhs.todense()  # pyright: ignore
        # ignore NECESSITY - pyright can't tell that lhs/rhs are of type sparray

    return np.sum((mannwhitneyu(lhs, rhs, axis=0).pvalue * ngenes) < 0.05)


def score_deg(
    X: MatrixLike,
    W: BoolArr,
    deg_cutoff: float = 1.5,
) -> Integral:
    """
    implements a heuristic to score label adjustment based on number of DEGs produced by new labels.

    :param X: input cell by gene matrix
    :param V: 1-d boolean array of condition labels (True corresponds to case, False to control)
    :param W: 1-d boolean array of adjusted condition labels (True correspons to case, False to control)
    :return: number of DEGs between conditions (as determined by a bonferroni-corrected mann-whitney U test p value of less than 0.05 and an absolute log2 fold change of more than 1.5)
    """

    lhs, rhs = X[~W, :], X[W, :]
    return mannwhitneyu_ndeg(
        lhs,  # pyright: ignore
        rhs,  # pyright: ignore
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

    1) case cells labelled affected vs case cells labelled unaffected (difference should be as large as possible)
    2) control cells vs case cells labelled unaffected (difference should be as small as possible)

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
            case_only_expr[~case_only_vhat],  # pyright: ignore
            # X values from labeled affected
            case_only_expr[case_only_vhat],  # pyright: ignore
            deg_cutoff,
        )
        * score_weight_relab
    ) - (
        mannwhitneyu_ndeg(
            # X values from true control
            ctl_only_expr,  # pyright: ignore
            # X values from labeled unaffected
            case_only_expr[~case_only_vhat],  # pyright: ignore
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
) -> Real:
    """
    implements a heuristic to score label adjustment based on the distribution difference, using the Kolmogorov-Smirnov (or KS) statistic, between p_hat values.
    specifically two comparisons are run, and the return value is a weighted difference of:

    1) case cells labelled affected vs case cells labelled unaffected (difference should be as large as possible)
    2) control cells vs case cells labelled unaffected (difference should be as small as possible)

    :param Y: 1-d float array of condition scores
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param W: boolean array of adjusted condition labels (False corresponds to control, True to case)
    :param score_weight_relab: weight given to KS stat of first comparison
    :param score_weight_vsctl: weight given to KS stat of second comparison
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
            # ks_2samp alternative refers to the CDFs of data1/data2
            # the trend in which is inversely related to the trend of
            # the means of data1/data2. as such, since our alternative
            # hypothesis is that labeled unaffected cells have a p_hat
            # distribution less than that of labeled affected cells,
            # the alternative argument is set to "greater", as
            # the trend for their respective CDFs would be inverted
        ).statistic  # pyright: ignore
        * score_weight_relab
    ) - (
        ks_2samp(
            # Y values from true control
            ctl_only_pred,
            # Y values from labeled unaffected
            case_only_pred[~case_only_vhat],
        ).statistic  # pyright: ignore
        * score_weight_vsctl
    )
    # ignore NECESSITY - scipy.stats.ks_2samp lacking proper
    # annotation on return type (currently set to _)


@dataclass
class NaiveMinScoreTuner(Tuner):
    """
    tuner class which attempts to select for an optimal k by selecting the k with maximal
    score as calculated by ``self.score_fn`` for each provided k in ``self.k_range``

    assumes that provided pipeline takes an argument ``k`` at invocation
    which dictates the number of dimensions to calculate in the reduction

    :param score_fn: function which given some arguments is able to determine a "score" for the pipeline results
    :param k_range: iterable collection of dimensions over which pipeline should be run to determine optimal k
    """

    score_fn: Callable[..., Real]
    k_range: Iterable[int]

    def __call__(self, algo: Pipeline, **kwargs) -> tuple[int, Mapping[int, tuple[NumArr, BoolArr, Real]]]:
        k_range = list(self.k_range)

        # run an explicit check before starting pipeline to avoid unnecessary work
        # when calling algo.dimr_fn outside of Pipeline.__call__, which skips "type-checking"
        for k in k_range:
            algo.check(kwargs | {"k": k})

        if algo.cachable_dimr:
            fn = algo.dimr_fn
            cache = wcall(
                kwargs | {"k": max(k_range)},
                out_to_dict(fn, getattr(fn, _INTERNAL_WRAP_ATTR_NAME)),
                algo.strict,
            )["Z"]

            algo = algo.update(dimr_fn=lambda k: cache[:, :k])

        res = dict()
        for k in sorted(k_range, reverse=True):
            _, _, w = algo(k=k, **kwargs)
            score = wcall(w, self.score_fn, algo.strict)
            res[k] = (w["Y"], w["W"], score)

        keep = max(sorted(res.items(), key=lambda t: t[0]), key=lambda t: t[1][2])[0]

        return keep, res


@dataclass
class FixPointTuner[T: float](Tuner):
    """
    tuner class which attempts to select for an optimal k (k_opt) > ``self.start_k`` such that
    where:

    - v_hat(x, k): is the adjusted label for a cell x under dimensionality reduction of dimension k

    - n_change(k): is # of cells x for which it is true that v_hat(x, k) != v_hat(x, k-1)

    for all k in (k_opt-``self.min_stable``, k_opt]: n_change(k) < # of cells * ``self.pct_delta``
    (e.g. first dimension g.t. ``self.start_k`` such that for previous ``self.min_stable`` dimensions, labels haven't changed by more than ``self.pct_delta`` percent)

    if ``self.force_dec`` is ``True``, we also enforce the property that:
    for all k in (k_opt-``self.min_stable``, k_opt]: n_change(k) <= n_change(k-1)
    (e.g. number of changing labels was decreasing for all ``self.min_stable`` k prior to selected optimal k)

    assumes that provided pipeline takes an argument ``k`` at invocation
    which dictates the number of dimensions to calculate in the reduction

    :param min_stable: number of dimensions for which stability property outlined above must hold
    :param start_k: number of dimensions to start iteration process with
    :param pct_delta: percent change in label considered allowable
    :param force_dec: whether to also enforce decreasing property outlined above, defaults to ``False``
    """

    min_stable: int
    start_k: int
    pct_delta: float
    force_dec: bool = False

    def __post_init__(self):
        assert 0 <= self.pct_delta < 1.0, f"self.allowable_change must be a float in [0, 1), but got {self.pct_delta}"

    def __call__(self, algo: Pipeline, **kwargs) -> tuple[int, Mapping[int, tuple[NumArr, BoolArr, T]]]:
        if algo.cachable_dimr:
            fn = algo.dimr_fn

            run_dimr = lambda k: wcall(  # noqa: E731
                kwargs | {"k": k},
                out_to_dict(fn, getattr(fn, _INTERNAL_WRAP_ATTR_NAME)),
                algo.strict,
            )["Z"]

            cache = run_dimr(self.start_k + self.min_stable)

            def dimr(k: int) -> FloatMtx:
                nonlocal cache
                if k > cache.shape[1]:
                    cache = run_dimr(k + self.min_stable)
                return cache[:, k]

            algo = algo.update(dimr_fn=dimr)

        res = dict()
        k = self.start_k
        tick = 0
        pct_change = 1.0
        y_hat, v_hat, _ = algo(k=k, **kwargs)

        while tick < self.min_stable:
            res[k] = (y_hat, v_hat, pct_change)
            k += 1

            y_hat, v_hat, _ = algo(k=k, **kwargs)
            pct_change = np.sum(res[k - 1][1] != v_hat) / len(v_hat)

            if pct_change < self.pct_delta and ((not self.force_dec) or pct_change <= res[k - 1][2]):
                tick += 1
            else:
                tick = 0

        res[k] = (y_hat, v_hat, pct_change)

        return k, res
