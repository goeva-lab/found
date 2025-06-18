from dataclasses import dataclass
from inspect import Signature, signature
from numbers import Real
from types import UnionType, resolve_bases
from typing import Any, Callable, Iterable, Self, Union, get_args

import anndata as ad

from .types import (
    BoolArr,
    FloatMtx,
    MatrixLike,
    NumArr,
)


# TODO: determine if there's a better way to do this
def strip_generic(tp: Any) -> Any:

    # recursive case for handling UnionType
    if isinstance(tp, UnionType):
        return Union[*map(strip_generic, get_args(tp))]

    return resolve_bases([tp])[0]


def wcall[T](w: dict, fn: Callable[..., T], strict: bool) -> T:
    kwargs = {}
    for k, v in signature(fn).parameters.items():

        # TODO: determine if type-checking strictness should be relaxable
        if strict and (v.annotation != Signature.empty) and (not isinstance(w[k], strip_generic(v.annotation))):
            raise TypeError(
                f"function: `{fn.__name__}` expected value of type "
                f"`{v.annotation}` to be provided for argument `{k}`, but "
                f"value `{w[k]}` of type {type(w[k])} was provided instead"
            )

        kwargs[k] = w[k]

    return fn(**kwargs)


@dataclass
class Pipeline:
    """
    constructor for a pipeline which provided with individual pipeline components (see class fields),
    creates a pipeline for detecting and re-classifying cells affected/unaffected by the case condition.

    Functions are run in the following order: norm_fn -> dimr_fn -> regr_fn -> binr_fn -> scor_fn

    :param norm_fn: normalization/transformation function (output accessible to further functions via a parameter named N)
    :param dimr_fn: dimensionality reduction function (output accessible to further functions via a parameter named Z)
    :param regr_fn: regression function (output accessible to further functions via a parameter named Y)
    :param binr_fn: binarization function (output accessible to further functions via a parameter named W)
    :param scor_fn: scoring function
    :param cachable_dimr: boolean indicating if the first k dimensions of the output of dimr_fn stable when requesting larger dimensions (e.g. is ``dimr_fn(X, k) = dimr_fn(X, k+n)[:, :k]``); caution with setting this without care as it can lead to incorrect results
    :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious TypeError failures
    """

    norm_fn: Callable[..., MatrixLike]
    dimr_fn: Callable[..., FloatMtx]
    regr_fn: Callable[..., NumArr]
    binr_fn: Callable[..., BoolArr]
    scor_fn: Callable[..., Real]
    cachable_dimr: bool = False
    strict: bool = True

    def __call__(self, **kwargs) -> tuple[NumArr, BoolArr, Real]:
        w = kwargs
        w["N"] = wcall(w, self.norm_fn, self.strict)
        w["Z"] = wcall(w, self.dimr_fn, self.strict)
        w["Y"] = wcall(w, self.regr_fn, self.strict)
        w["W"] = wcall(w, self.binr_fn, self.strict)
        score = wcall(w, self.scor_fn, self.strict)

        return w["Y"], w["W"], score

    def update(
        self,
        norm_fn: Callable[..., MatrixLike] | None = None,
        dimr_fn: Callable[..., FloatMtx] | None = None,
        regr_fn: Callable[..., NumArr] | None = None,
        binr_fn: Callable[..., BoolArr] | None = None,
        scor_fn: Callable[..., Real] | None = None,
        cachable_dimr: bool | None = None,
        strict: bool | None = None,
    ) -> Self:
        """
        convenience constructor for creating a new :py:class:`~found.adapters.Pipeline` from an existing one (does not mutate original object).

        :param norm_fn: normalization/transformation function (output accessible to further functions via a parameter named N)
        :param dimr_fn: dimensionality reduction function (output accessible to further functions via a parameter named Z)
        :param regr_fn: regression function (output accessible to further functions via a parameter named Y)
        :param binr_fn: binarization function (output accessible to further functions via a parameter named W)
        :param scor_fn: scoring function
        :param cachable_dimr: boolean indicating if the first k dimensions of the output of dimr_fn stable when requesting larger dimensions (e.g. is ``dimr_fn(X, k) = dimr_fn(X, k+n)[:, :k]``); caution with setting this without care as it can lead to incorrect results
        :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious TypeError failures
        """
        return type(self)(
            norm_fn=self.norm_fn if norm_fn is None else norm_fn,
            dimr_fn=self.dimr_fn if dimr_fn is None else dimr_fn,
            regr_fn=self.regr_fn if regr_fn is None else regr_fn,
            binr_fn=self.binr_fn if binr_fn is None else binr_fn,
            scor_fn=self.scor_fn if scor_fn is None else scor_fn,
            cachable_dimr=self.cachable_dimr if cachable_dimr is None else cachable_dimr,
            strict=self.strict if strict is None else strict,
        )

    @classmethod
    def from_proc_ad(
        cls,
        norm_key: str | None,
        dimr_key: str,
        regr_fn: Callable[..., NumArr],
        binr_fn: Callable[..., BoolArr],
        scor_fn: Callable[..., Real],
    ) -> Self:
        """
        convenience constructor for creating :py:class:`~found.adapters.Pipeline`\\ s from processed :py:class:`~anndata.AnnData` objects with already computed count transformation and dimensionality reduction.

        resulting :py:class:`~found.adapters.Pipeline` must be called with a named argument ``adata`` providing :py:class:`~anndata.AnnData` object.
        provided ``regr_fn``, ``binr_fn``, ``scor_fn`` methods should receive :py:class:`~anndata.AnnData` object via their ``adata`` argument.

        resulting :py:class:`~found.adapters.Pipeline` can optionally be called with a named argument ``k`` specifying the desired dimensionality of the dimensionality reduction space, otherwise the full provided space will be used.

        :param norm_key: key for ``.layers`` slot specifiying normalized counts, if set to None, ``.X`` slot is used instead
        :param dimr_key: key for ``.obsm`` slot specifying dimensionality reduction matrix
        :param regr_fn: regression function (output accessible to further functions via a parameter named Y)
        :param binr_fn: binarization function (output accessible to further functions via a parameter named W)
        :param scor_fn: scoring function
        """

        def norm_fn(adata: ad.AnnData) -> MatrixLike:
            if norm_key is None:
                norm = adata.X
                msg = f"expected adata.X to be of type {MatrixLike}, got {type(norm)} instead"
            else:
                norm = adata.layers[norm_key]
                msg = f"expected adata.X to be of type {MatrixLike}, got {type(norm)} instead"
            assert isinstance(norm, strip_generic(MatrixLike)), msg
            return norm  # pyright: ignore
            # ignore NECESSITY - isinstance check through strip_generic not understood by type checker

        def dimr_fn(adata: ad.AnnData, k: int | None = None) -> FloatMtx:
            dimr: FloatMtx = adata.obsm[dimr_key]  # pyright: ignore
            # ignore NECESSITY - isinstance check through strip_generic not understood by type checker
            assert isinstance(
                dimr, strip_generic(FloatMtx)
            ), f'expected adata.obsm["{dimr}"] to be of type {FloatMtx}, got {type(dimr)} instead'
            if k is not None:
                assert (
                    dimr.shape[1] >= k
                ), f'provided dimensionality reduction matrix in adata.obsm["{dimr}"] is of shape {dimr.shape}, but a {k}-d space was queried'
            return dimr[:, :k]  # pyright: ignore
            # ignore NECESSITY - np indexing does not preserve array shape
            # `:`-index keeps first dimension, and `:k`-index also preserves second dimension

        return cls(norm_fn, dimr_fn, regr_fn, binr_fn, scor_fn, True)


def heuristic_loop(
    algo: Pipeline,
    k_range: Iterable[int],
    **kwargs,
) -> tuple[NumArr, BoolArr, int]:
    """
    runs HiDDEN on a given AnnData object, trying to optimize for a specific set of dimensions specified by k_range.

    :param algo: algorithm pipeline (expected to use parameter ``X`` as counts matrix, ``V`` as original condition annotation, and ``k`` as number of dimensions to reduce to)
    :param k_range: range of dimensionality reduction dimensions to iterate over when trying to find best value
    :param kwargs: additional variables to pass into pipeline
    :return: 3-tuple consisting of:

        - 1-d array of prediction outputs by model
        - binarized labels from prediction values
        - optimal number of dimensions selected
    """

    # cache normalization values between loops
    norm = wcall(kwargs, algo.norm_fn, algo.strict)
    algo = algo.update(norm_fn=lambda: norm)

    if algo.cachable_dimr:
        k_range = list(k_range)
        fn = algo.dimr_fn
        cache = wcall(kwargs | {"N": norm, "k": max(k_range)}, fn, algo.strict)

        algo = algo.update(
            dimr_fn=lambda k: cache[:, :k],  # pyright: ignore
            # ignore NECESSITY - np indexing does not preserve array shape
            # `:`-index keeps first dimension, and `:k`-index also preserves second dimension
            # meaning output will be 2-d array given that cache is a 2-d array
        )

    res: dict[int, tuple[tuple[NumArr, BoolArr], Real]] = {}
    for k in k_range:
        y_hat, v_hat, score = algo(k=k, **kwargs)
        res[k] = ((y_hat, v_hat), score)

    # sort items by k to ensure smallest dimensions
    # are first, and since max returns first maximal
    # value, smallest k of maximal score will be returned
    keep = max(
        sorted(res.items(), key=lambda t: t[0]),
        key=lambda t: t[1][1],
    )
    return *keep[1][0], keep[0]
