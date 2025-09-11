from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .adapters import _INTERNAL_WRAP_ATTR_NAME, Pipeline, out_to_dict, wcall
from .types import BoolArr, FloatMtx, MatrixLike, NumArr, NumericScalar, check_sequence, strip_generic


class Tuner[HyperparameterType, ScoreType](ABC):
    """
    abstract base class for a hyper parameter tuner which wraps some pipeline
    """

    @abstractmethod
    def __call__(
        self, algo: Pipeline, **kwargs
    ) -> tuple[HyperparameterType, Mapping[HyperparameterType, tuple[NumArr, BoolArr, ScoreType]]]: ...


@dataclass(frozen=True)
class NaiveMaxScoreTuner(Tuner):
    """
    tuner class which attempts to select for an optimal k by selecting the k with maximal
    score as calculated by ``self.score_fn`` for each provided k in ``self.k_range``

    assumes that provided pipeline takes an argument ``k`` at invocation
    which dictates the number of dimensions to calculate in the reduction

    :param score_fn: function which given some arguments is able to determine a "score" for the pipeline results
    :param k_range: iterable collection of dimensions over which pipeline should be run to determine optimal k
    """

    score_fn: Callable[..., NumericScalar]
    k_range: Iterable[int]

    def __call__(self, algo: Pipeline, **kwargs) -> tuple[int, Mapping[int, tuple[NumArr, BoolArr, NumericScalar]]]:
        k_range = list(self.k_range)

        # run an explicit check before starting pipeline to avoid unnecessary work
        # when calling algo.dimr_fn outside of Pipeline.__call__, which skips "type-checking"
        for k in k_range:
            check_sequence(
                [
                    (algo.dimr_fn, getattr(algo.dimr_fn, _INTERNAL_WRAP_ATTR_NAME)),
                    (algo.regr_fn, getattr(algo.regr_fn, _INTERNAL_WRAP_ATTR_NAME)),
                    (algo.binr_fn, getattr(algo.binr_fn, _INTERNAL_WRAP_ATTR_NAME)),
                    (self.score_fn, ("score",)),
                ],
                algo.strict,
                kwargs | {"k": k},
            )

        if algo.cachable_dimr:
            fn = algo.dimr_fn
            cache = wcall(
                kwargs | {"k": max(k_range)},
                out_to_dict(fn, getattr(fn, _INTERNAL_WRAP_ATTR_NAME)),
                algo.strict,
            )["Z"]
            assert isinstance(cache, strip_generic(MatrixLike))

            def dimr(k: int) -> FloatMtx:
                nonlocal cache
                return cache[:, :k]  # pyright: ignore[reportIndexIssue]

            algo = algo.update(dimr_fn=dimr)

        res = dict()
        for k in sorted(k_range, reverse=True):
            _, _, w = algo(k=k, **kwargs)
            score = wcall(w, self.score_fn, algo.strict)
            res[k] = (w["Y"], w["W"], score)

        keep = max(sorted(res.items(), key=lambda t: t[0]), key=lambda t: t[1][2])[0]

        return keep, res


@dataclass(frozen=True)
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
            # run explicit type check on pipeline necessary pre-update since
            # cached dimr construction hides real function dependencies in captured variables
            algo.check(kwargs | {"k": self.start_k})

            fn = algo.dimr_fn

            cache = np.ndarray(shape=(0, 0))

            def dimr(k: int) -> FloatMtx:
                nonlocal cache
                if k > cache.shape[1]:  # pyright: ignore[reportAttributeAccessIssue]
                    cache = wcall(
                        kwargs | {"k": k + self.min_stable},
                        out_to_dict(fn, getattr(fn, _INTERNAL_WRAP_ATTR_NAME)),
                        algo.strict,
                    )["Z"]
                return cache[:, :k]  # pyright: ignore[reportReturnType, reportIndexIssue]

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
