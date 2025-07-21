from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import log2
from numbers import Real
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

from .adapters import _INTERNAL_WRAP_ATTR_NAME, Pipeline, out_to_dict, wcall
from .types import BoolArr, MatrixLike, NumArr


class Tuner[HyperparameterType, ScoreType](ABC):
    """
    abstract base class for a hyper parameter tuner which wraps some pipeline
    """

    @abstractmethod
    def __call__(
        self, algo: Pipeline, **kwargs
    ) -> tuple[HyperparameterType, Mapping[HyperparameterType, tuple[NumArr, BoolArr, ScoreType]]]: ...


def score_deg(X: MatrixLike, W: BoolArr, deg_cutoff: float = 1.5) -> Real:
    """
    implements a heuristic to score label adjustment based on number of DEGs produced by new labels.

    :param X: input cell by gene matrix
    :param V: 1-d boolean array of condition labels (True corresponds to case, False to control)
    :param W: 1-d boolean array of adjusted condition labels (True correspons to case, False to control)
    :return: number of DEGs between conditions (as determined by a bonferroni-corrected mann-whitney U test p value of less than 0.05 and an absolute log2 fold change of more than 1.5)
    """

    found_ctrl, found_case = X[~W, :], X[W, :]

    ngenes = X.shape[1]  # pyright: ignore
    # ignore NECESSITY - spmatrix.shape is not annotated
    # to be a tuple, inferred annotations set it to None

    if isinstance(X, np.ndarray):
        ctrl_mean, case_mean = np.mean(found_ctrl, axis=0), np.mean(found_case, axis=0)
    else:
        ctrl_mean, case_mean = found_ctrl.mean(axis=0), found_case.mean(axis=0)

    # remove genes for which log2fc is not sensical
    # (i.e. mean of zero in either condition)
    gtz = np.logical_and(ctrl_mean > 0, case_mean > 0)
    # remove genes where log2FC is less than 1.5
    lfc = np.abs(np.log2(case_mean[gtz] / ctrl_mean[gtz])) > deg_cutoff

    found_ctrl, found_case = found_ctrl[:, gtz][:, lfc], found_case[:, gtz][:, lfc]

    # mannwhitneyu does not work on sparse arrays
    if not isinstance(X, np.ndarray):
        found_case, found_ctrl = found_case.todense(), found_ctrl.todense()  # pyright: ignore
        # ignore NECESSITY - if X is not np.ndarray, it is a sparse array, meaning
        # so are affected and unaffected, but pyright can't detect this currently

    return np.sum((mannwhitneyu(found_case, found_ctrl, axis=0).pvalue * ngenes) < 0.05)


def score_phatdiff(Y: NumArr, V: BoolArr, W: BoolArr) -> Real:
    """
    implements a heuristic to score label adjustment based on the distribution difference between p_hat values in the case condition.

    :param Y: 1-d float array of condition scores
    :param V: 1-d boolean array of condition labels (False corresponds to control, True to case)
    :param W: boolean array of adjusted condition labels (False corresponds to control, True to case)
    :return: -log2 of the Kolmogorov-Smirnov p value value between ``Y`` values for unaffected vs affected in the originally labelled case condition.
    """
    case_only_pred = Y[V]
    case_only_vhat = W[V]
    return -log2(
        max(
            ks_2samp(
                case_only_pred[~case_only_vhat],  # Y values from labeled unaffected
                case_only_pred[case_only_vhat],  #  Y values from labeled affected
                alternative="greater",
                # ks_2samp alternative refers to the CDFs of data1/data2
                # the trend in which is inversely related to the trend of
                # the means of data1/data2. as such, since our alternative
                # hypothesis is that labeled unaffected cells have a p_hat
                # distribution less than that of labeled affected cells,
                # the alternative argument is set to "greater", as
                # the trend for their respective CDFs would be inverted
            ).pvalue,  # pyright:ignore
            2**-1000,
        )
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
        algo.check({k: type(v) for k, v in kwargs.items()} | {"k": int})
        k_range = list(self.k_range)

        if algo.cachable_dimr:
            fn = algo.dimr_fn
            cache = wcall(
                kwargs | {"k": max(k_range)},
                out_to_dict(fn, getattr(fn, _INTERNAL_WRAP_ATTR_NAME)),
                algo.strict,
            )["Z"]

            algo = algo.update(
                dimr_fn=lambda k: cache[:, :k],  # pyright: ignore
                # ignore NECESSITY - np indexing does not preserve array shape
                # `:`-index keeps first dimension, and `:k`-index also preserves second dimension
                # meaning output will be 2-d array given that cache is a 2-d array
            )

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
        algo.check({k: type(v) for k, v in kwargs.items()} | {"k": int})

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
