from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature
from inspect import _empty as iempty
from types import UnionType, resolve_bases
from typing import Any, Literal, Protocol, Self, Union, get_args, get_origin

import anndata as ad
import numpy as np

from .types import BoolArr, FloatMtx, NumArr

_INTERNAL_WRAP_ATTR_NAME = "__FOUND_WRAPPED_INTERNAL_NAMES"


# @TODO: determine if there's a better way to do this
def strip_generic(tp: Any) -> Any:
    # recursive case for handling UnionType
    if isinstance(tp, UnionType):
        return Union[*map(strip_generic, get_args(tp))]

    return resolve_bases([tp])[0]


def check(w: dict[str, Any], p: Parameter, name: str):
    if p.annotation is not iempty:
        if get_origin(p.annotation) is Literal:
            chk = w[p.name] in get_args(p.annotation)
        else:
            chk = isinstance(w[p.name], strip_generic(p.annotation))
        if not chk:
            raise TypeError(
                f"function: `{name}` expected value of type "
                f"`{p.annotation}` to be provided for argument `{p.name}`, but "
                f"value {w[p.name]} of type {type(w[p.name])} was provided instead"
            )
        return chk


def wcall[T](w: dict[str, Any], fn: Callable[..., T], strict: bool) -> T:
    kwargs = dict()
    for p in signature(fn).parameters.values():
        if p.name in w:
            if strict:
                check(w, p, fn.__name__)
            kwargs[p.name] = w[p.name]

    return fn(**kwargs)


def out_to_dict[*I, O](func: Callable[[*I], O], out_names: tuple[str, ...]) -> Callable[[*I], dict[str, Any]]:
    @wraps(func)
    def w(*args: *I, **kwargs) -> dict[str, Any]:
        o = func(*args, **kwargs)
        o = o if isinstance(o, tuple) else (o,)
        assert len(o) == len(out_names), (
            f"function {func.__name__} was decorated with named outputs {out_names} "
            f"of length {len(out_names)}, but returned {o}, of length {len(out_names)}"
        )
        return {k: v for k, v in zip(out_names, o)}

    return w


def step_fn[Fn: Callable](*out_names: str) -> Callable[[Fn], Fn]:
    def _(func: Fn) -> Fn:
        # lazy way to create a copy of func s.t. setattr doesn't modify func
        @wraps(func)
        def c(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(c, _INTERNAL_WRAP_ATTR_NAME, out_names)

        return c  # pyright: ignore[reportReturnType]
        # ignore NECESSITY - from the definition of c, we can
        # see that it will replicate the type signature of func

    return _


@dataclass(frozen=True)
class Pipeline:
    """
    class which when provided with individual pipeline components (see fields),
    creates a pipeline for detecting and re-classifying cells affected/unaffected by the case condition.

    functions are run in the following order: dimr_fn -> regr_fn -> binr_fn

    :param dimr_fn: dimensionality reduction function (output accessible to further functions via a parameter named Z unless explicitly wrapped by step_fn)
    :param regr_fn: regression function (output accessible to further functions via a parameter named Y unless explicitly wrapped by step_fn)
    :param binr_fn: binarization function (output accessible to further functions via a parameter named W unless explicitly wrapped by step_fn)

    :param cachable_dimr: boolean indicating if the first k dimensions of the output of dimr_fn stable when requesting larger dimensions (e.g. is ``dimr_fn(X, k) = dimr_fn(X, k+n)[:, :k]``); caution with setting this without care as it can lead to incorrect results
    :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious TypeError failures
    """

    dimr_fn: Callable
    regr_fn: Callable
    binr_fn: Callable
    cachable_dimr: bool = False
    strict: bool = True

    def check(self, w: dict[str, Any]):
        """
        convenience function, given a set of provided arguments, validates if the pipeline can run
        returns if yes, raises :py:class:`~ValueError` if no

        :param w: dictionary of initial pipeline variables
        """

        added = set()

        for fn, out in [
            (self.dimr_fn, set(getattr(self.dimr_fn, _INTERNAL_WRAP_ATTR_NAME))),
            (self.regr_fn, set(getattr(self.regr_fn, _INTERNAL_WRAP_ATTR_NAME))),
            (self.binr_fn, set(getattr(self.binr_fn, _INTERNAL_WRAP_ATTR_NAME))),
        ]:
            for p in signature(fn).parameters.values():
                if (p.default is iempty) and (p.name not in (w.keys() | added)):
                    raise ValueError(
                        f"pipeline called with missing arguments, provided function {fn.__name__} expects presence of {p.name}, but it was not provided"
                    )
                if self.strict and (p.name not in added) and (p.name in w):
                    check(w, p, fn.__name__)
            added = added.union(out)

    def __post_init__(self):
        if not (hasattr(self.dimr_fn, _INTERNAL_WRAP_ATTR_NAME)):
            object.__setattr__(self, "dimr_fn", step_fn("Z")(self.dimr_fn))

        if not hasattr(self.regr_fn, _INTERNAL_WRAP_ATTR_NAME):
            object.__setattr__(self, "regr_fn", step_fn("Y")(self.regr_fn))
        elif "Y" not in (n := getattr(self.regr_fn, _INTERNAL_WRAP_ATTR_NAME)):
            raise ValueError(f"expected wrapped self.regr_fn to a named output variable named `Y`, but got names {n}")

        if not hasattr(self.binr_fn, _INTERNAL_WRAP_ATTR_NAME):
            object.__setattr__(self, "binr_fn", step_fn("W")(self.binr_fn))
        elif "W" not in (n := getattr(self.binr_fn, _INTERNAL_WRAP_ATTR_NAME)):
            raise ValueError(f"expected wrapped self.binr_fn to a named output variable named `W`, but got names {n}")

    def __call__(self, **kwargs) -> tuple[NumArr, BoolArr, dict[str, Any]]:
        w = kwargs

        self.check(w)

        w |= wcall(w, out_to_dict(self.dimr_fn, getattr(self.dimr_fn, _INTERNAL_WRAP_ATTR_NAME)), self.strict)
        w |= wcall(w, out_to_dict(self.regr_fn, getattr(self.regr_fn, _INTERNAL_WRAP_ATTR_NAME)), self.strict)
        w |= wcall(w, out_to_dict(self.binr_fn, getattr(self.binr_fn, _INTERNAL_WRAP_ATTR_NAME)), self.strict)

        return w["Y"], w["W"], w

    def update(
        self,
        dimr_fn: Callable | None = None,
        regr_fn: Callable | None = None,
        binr_fn: Callable | None = None,
        cachable_dimr: bool | None = None,
        strict: bool | None = None,
    ) -> Self:
        """
        convenience constructor for creating a new :py:class:`~found.adapters.Pipeline` from an existing one (does not mutate original object).

        :param norm_fn: normalization/transformation function (output accessible to further functions via a parameter named N)
        :param dimr_fn: dimensionality reduction function (output accessible to further functions via a parameter named Z)
        :param regr_fn: regression function (output accessible to further functions via a parameter named Y)
        :param binr_fn: binarization function (output accessible to further functions via a parameter named W)
        :param cachable_dimr: boolean indicating if the first k dimensions of the output of dimr_fn stable when requesting larger dimensions (e.g. is ``dimr_fn(X, k) = dimr_fn(X, k+n)[:, :k]``); caution with setting this without care as it can lead to incorrect results
        :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious TypeError failures
        """
        return type(self)(
            dimr_fn=self.dimr_fn if dimr_fn is None else dimr_fn,
            regr_fn=self.regr_fn if regr_fn is None else regr_fn,
            binr_fn=self.binr_fn if binr_fn is None else binr_fn,
            cachable_dimr=self.cachable_dimr if cachable_dimr is None else cachable_dimr,
            strict=self.strict if strict is None else strict,
        )

    @classmethod
    def from_proc_ad(
        cls,
        dimr_key: str,
        regr_fn: Callable,
        binr_fn: Callable,
        strict: bool | None = None,
    ) -> Self:
        """
        convenience constructor for creating :py:class:`~found.adapters.Pipeline`\\ s from processed :py:class:`~anndata.AnnData` objects with already computed count transformation and dimensionality reduction.

        resulting :py:class:`~found.adapters.Pipeline` must be called with a named argument ``adata`` providing :py:class:`~anndata.AnnData` object.
        provided ``regr_fn``, ``binr_fn``, ``scor_fn`` methods should receive :py:class:`~anndata.AnnData` object via their ``adata`` argument.

        resulting :py:class:`~found.adapters.Pipeline` can optionally be called with a named argument ``k`` specifying the desired dimensionality of the dimensionality reduction space, otherwise the full provided space will be used.

        :param dimr_key: key for ``.obsm`` slot specifying dimensionality reduction matrix
        :param regr_fn: regression function (output accessible to further functions via a parameter named Y)
        :param binr_fn: binarization function (output accessible to further functions via a parameter named W)
        :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious TypeError failures, defaults to Pipeline.strict default
        """

        def dimr_fn(adata: ad.AnnData, k: int | None = None) -> FloatMtx:
            dimr: FloatMtx = adata.obsm[dimr_key]  # pyright: ignore[reportAssignmentType]
            # ignore NECESSITY - isinstance check through strip_generic not understood by type checker
            assert isinstance(dimr, strip_generic(FloatMtx)), (
                f'expected adata.obsm["{dimr}"] to be of type {FloatMtx}, got {type(dimr)} instead'
            )
            if k is not None:
                assert dimr.shape[1] >= k, (
                    f'provided dimensionality reduction matrix in adata.obsm["{dimr}"] is of shape {dimr.shape}, but a {k}-d space was queried'
                )
            return dimr[:, :k]  # pyright: ignore[reportReturnType]
            # ignore NECESSITY - numpy indexing does not preserve array shape
            # `:`-index keeps first dimension, and `:k`-index also preserves second dimension

        return cls(dimr_fn, regr_fn, binr_fn, True, *([] if strict is None else [strict]))


# create protocol since Callable does not allow specifying keyword only arguments
class GroupbyOut[T, G](Protocol):
    def __call__(self, grp: G, /, **kwargs) -> T: ...


def wrap_gby_fn[T, G](
    fn: Callable[..., T], which_args: Collection[str], grps: Mapping[G, np.ndarray[tuple[int], np.dtype[np.integer]]]
) -> GroupbyOut[T, G]:
    def f(grp: G, /, **kwargs) -> T:
        new_args = {k: kwargs[k][grps[grp]] for k in which_args}

        for k in new_args:
            # materialize anndata view before it is repeatedly accessed downstream
            if isinstance(new_args[k], ad.AnnData):
                new_args[k] = new_args[k].copy()

            # more specializations here

        return fn(**(kwargs | new_args))

    return f
