from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Protocol, Self

import anndata as ad
import numpy as np

from .types import BoolArr, FloatMtx, NumArr, check_sequence, strip_generic, wcall

_INTERNAL_WRAP_ATTR_NAME = "__FOUND_WRAPPED_INTERNAL_NAMES"


def out_to_dict[*I](func: Callable[[*I], tuple], out_names: tuple[str, ...]) -> Callable[[*I], dict[str, object]]:
    @wraps(func)
    def w(*args: *I, **kwargs) -> dict[str, object]:
        o = func(*args, **kwargs)
        assert len(o) == len(out_names), (
            f"function {getattr(func, '__name__', '[NO NAME FOUND]')} was decorated with named outputs {out_names} "
            f"of length {len(out_names)}, but returned {o}, of length {len(out_names)}"
        )
        return {k: v for k, v in zip(out_names, o)}

    return w


def step_fn[Fn: Callable[..., tuple]](*out_names: str) -> Callable[[Fn], Fn]:
    """
    decorator for functions that return tuples, returns a copy which "annotates" individual output values with names for use in :py:class:`~found.adapters.Pipeline`.

    returned function behaves identically to the original un-decorated function, the only modification is the addition of a private attribute to the function object.

    :param out_names: desired "names" of return values, in the same order as they are in the returned tuple.

    :returns: named return value annotated function
    """

    def _(func: Fn) -> Fn:
        # lazy way to create a copy of func s.t. setattr doesn't modify func
        @wraps(func)
        def c(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(c, _INTERNAL_WRAP_ATTR_NAME, out_names)

        return c  # ty:ignore[invalid-return-type]
        # ignore NECESSITY - from the definition of c, we can
        # see that it will replicate the type signature of func

    return _


def wrap_to_ot[T](fn: Callable[..., T]) -> Callable[..., tuple[T]]:
    @wraps(fn)
    def w(*args, **kwargs) -> tuple[T]:
        return (fn(*args, **kwargs),)

    return w


@dataclass(frozen=True)
class Pipeline:
    """
    class which when provided with individual pipeline components (see fields),
    creates a pipeline for detecting and re-classifying cells affected/unaffected by the case condition.

    functions are run in the following order: ``dimr_fn`` → ``regr_fn`` → ``binr_fn``

    :param dimr_fn: dimensionality reduction function (output accessible to further functions via a parameter named Z unless explicitly wrapped by :py:func:`~found.adapters.step_fn`)
    :param regr_fn: regression function (output accessible to further functions via a parameter named Y unless explicitly wrapped by :py:func:`~found.adapters.step_fn`)
    :param binr_fn: binarization function (output accessible to further functions via a parameter named W unless explicitly wrapped by :py:func:`~found.adapters.step_fn`)

    :param cachable_dimr:
        boolean indicating if the first k dimensions of the output of dimr_fn stable when requesting larger dimensions (e.g. is ``dimr_fn(X, k) = dimr_fn(X, k+n)[:, :k]``)
        caution with setting this without care as it can lead to incorrect results when conducting hyperparameter optimization via :py:class:`~found.tune.Tuner`.
        as a rule of thumb, this property is generally only true for truncated PCA.

    :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious {py:exc}`~TypeError` failures
    """

    dimr_fn: Callable
    regr_fn: Callable
    binr_fn: Callable
    cachable_dimr: bool = False
    strict: bool = True

    def check(self, w: dict[str, object]):
        """
        convenience function, given a set of provided arguments, validates if the pipeline can run
        returns if yes, raises :py:exc:`~ValueError` if no

        :param w: dictionary of initial pipeline variables
        """

        check_sequence(
            [
                (self.dimr_fn, getattr(self.dimr_fn, _INTERNAL_WRAP_ATTR_NAME)),
                (self.regr_fn, getattr(self.regr_fn, _INTERNAL_WRAP_ATTR_NAME)),
                (self.binr_fn, getattr(self.binr_fn, _INTERNAL_WRAP_ATTR_NAME)),
            ],
            self.strict,
            w,
        )

    def __post_init__(self):
        if not (hasattr(self.dimr_fn, _INTERNAL_WRAP_ATTR_NAME)):
            object.__setattr__(self, "dimr_fn", step_fn("Z")(wrap_to_ot(self.dimr_fn)))

        if not hasattr(self.regr_fn, _INTERNAL_WRAP_ATTR_NAME):
            object.__setattr__(self, "regr_fn", step_fn("Y")(wrap_to_ot(self.regr_fn)))
        elif "Y" not in (n := getattr(self.regr_fn, _INTERNAL_WRAP_ATTR_NAME)):
            raise ValueError(f"expected wrapped self.regr_fn output to contain a named output variable `Y`, but got names {n}")

        if not hasattr(self.binr_fn, _INTERNAL_WRAP_ATTR_NAME):
            object.__setattr__(self, "binr_fn", step_fn("W")(wrap_to_ot(self.binr_fn)))
        elif "W" not in (n := getattr(self.binr_fn, _INTERNAL_WRAP_ATTR_NAME)):
            raise ValueError(f"expected wrapped self.binr_fn output to contain a named output variable `W`, but got names {n}")

    def __call__(self, **kwargs) -> tuple[NumArr, BoolArr, dict[str, object]]:
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

        :param dimr_fn: dimensionality reduction function (output accessible to further functions via a parameter named ``Z``)
        :param regr_fn: regression function (output accessible to further functions via a parameter named ``Y``)
        :param binr_fn: binarization function (output accessible to further functions via a parameter named ``W``)
        :param cachable_dimr:
            boolean indicating if the first k dimensions of the output of dimr_fn stable when requesting larger dimensions (e.g. is ``dimr_fn(X, k) = dimr_fn(X, k+n)[:, :k]``)
            caution with setting this without care as it can lead to incorrect results when conducting hyperparameter optimization via :py:class:`~found.tune.Tuner`.
            as a rule of thumb, this property is generally only true for truncated PCA.

        :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious :py:exc:`~TypeError` failures
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
        convenience constructor for creating a :py:class:`~found.adapters.Pipeline`\\ where the `dimr_fn` step fetches data from a specified :py:class:`~anndata.AnnData` :py:attr:`~anndata.AnnData.obsm` slot.

        resulting :py:class:`~found.adapters.Pipeline` must be called with a named argument ``adata`` providing an :py:class:`~anndata.AnnData` object.

        resulting :py:class:`~found.adapters.Pipeline` can optionally be called with a named argument ``k`` specifying the desired dimensionality of the dimensionality reduction space, otherwise the full provided space will be used.

        :param dimr_key: key for :py:attr:`~anndata.AnnData.obsm` slot specifying dimensionality reduction matrix
        :param regr_fn: regression function (output accessible to further functions via a parameter named Y)
        :param binr_fn: binarization function (output accessible to further functions via a parameter named W)
        :param strict: boolean indicating if the pipeline should conduct strict type checking, disable with caution if getting spurious :py:exc:`~TypeError` failures
        """

        def dimr_fn(adata: ad.AnnData, k: int | None = None) -> FloatMtx:
            dimr: FloatMtx = adata.obsm[dimr_key]
            assert isinstance(dimr, strip_generic(FloatMtx)), (
                f'expected adata.obsm["{dimr}"] to be of type {FloatMtx}, got {type(dimr)} instead'
            )
            if k is not None:
                assert dimr.shape[1] >= k, (
                    f'provided dimensionality reduction matrix in adata.obsm["{dimr}"] is of shape {dimr.shape}, but a {k}-d space was queried'
                )
            return dimr[:, :k]

        return cls(dimr_fn, regr_fn, binr_fn, True, *([] if strict is None else [strict]))


# create protocol since Callable does not allow specifying keyword only arguments
class GroupbyOut[T, G](Protocol):
    """:meta hide-value:"""

    def __call__(self, grp: G, /, **kwargs) -> T: ...


def wrap_gby_fn[T, G](
    fn: Callable[..., T],
    which_args: dict[str, Callable[[object, np.ndarray[tuple[int], np.dtype[np.integer]]], object]],
    grps: Mapping[G, np.ndarray[tuple[int], np.dtype[np.integer]]],
) -> GroupbyOut[T, G]:
    def f(grp: G, /, **kwargs) -> T:
        new_args = {k: v(kwargs[k], grps[grp]) for k, v in which_args.items()}

        for k in new_args:
            # materialize anndata view before it is repeatedly accessed downstream
            if isinstance(new_args[k], ad.AnnData):
                new_args[k] = new_args[k].copy()  # ty:ignore[unresolved-attribute]
                # ignore NECESSITY: new_args[k] check above ensures type is AnnData

            # more specializations here

        return fn(**(kwargs | new_args))

    return f
