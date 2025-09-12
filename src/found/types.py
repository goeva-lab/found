from collections.abc import Callable, Iterable
from inspect import _empty as iempty
from inspect import signature
from types import GenericAlias, UnionType
from typing import Any, Literal, Union, get_args, get_origin

import numpy as np
import scipy.sparse as sp

BoolArr = np.ndarray[tuple[int], np.dtype[np.bool_]]
NumArr = np.ndarray[tuple[int], np.dtype[np.number]]

FloatMtx = np.ndarray[tuple[int, int], np.dtype[np.floating]]
IntMtx = np.ndarray[tuple[int, int], np.dtype[np.integer]]
SparseMtx = sp.csr_array | sp.csc_array

MatrixLike = FloatMtx | IntMtx | SparseMtx

NumericScalar = int | float | np.floating | np.integer


# @TODO: determine if there's a better way to do this
def strip_generic(tp: type | UnionType | GenericAlias) -> type | UnionType:
    # recursive case for handling UnionType
    if isinstance(tp, UnionType):
        return Union[*map(strip_generic, get_args(tp))]

    if isinstance(tp, GenericAlias):
        if not isinstance(tp.__origin__, type | UnionType):
            raise ValueError(
                f"type {tp.__origin__} is unsupported for checking, please disable strict type checking or file an issue"
            )
        return tp.__origin__

    return tp


def wcall[T](w: dict[str, object], fn: Callable[..., T], strict: bool) -> T:
    kwargs = dict()
    for p in signature(fn).parameters.values():
        match p.kind:
            case p.KEYWORD_ONLY | p.POSITIONAL_OR_KEYWORD:
                if p.name in w:
                    if strict and p.annotation is not iempty and not vtype_check(w[p.name], p.annotation):
                        raise TypeError(
                            f"function: `{fn.__name__}` expected value of type "
                            f"`{p.annotation}` to be provided for argument `{p.name}`"
                            f", but value {w[p.name]} of type {type(w[p.name])} was provided instead"
                        )
                    kwargs[p.name] = w[p.name]
            case p.VAR_KEYWORD:
                kwargs |= w
            case _:
                raise TypeError(
                    f"function: `{fn.__name__}` has a strictly positional argument {p.name}"
                    f", which cannot be run through the pipeline mechanism of this library"
                )

    return fn(**kwargs)


def vtype_check(o: object, annot: type | UnionType | GenericAlias) -> bool:
    if get_origin(annot) is Literal:
        return o in get_args(annot)

    # @TODO: add explicit handling for isinstance-incompatible types (e.g. non-runtime-checkable Protocol)
    return isinstance(o, strip_generic(annot))


def ttype_check(t: type | UnionType | GenericAlias, annot: type | UnionType | GenericAlias) -> bool:
    # Any is subclass of all types
    if t is Any:
        return True
    # Literal cannot be checked against at type-level, so ignore and return True
    if get_origin(annot) is Literal:
        return True

    # @TODO: find ways to utilize generic annotations
    t = strip_generic(t)
    annot = strip_generic(annot)

    # @TODO: add explicit handling for isinstance-incompatible types (e.g. non-runtime-checkable Protocol)
    if isinstance(t, UnionType):
        return all(map(lambda t: issubclass(t, annot), get_args(t)))
    return issubclass(t, annot)


def check_sequence(seq: Iterable[tuple[Callable, tuple[str, *tuple[str, ...]]]], strict: bool, init_w: dict[str, object]):
    w_types: dict[str, tuple[object | None, None | (type | UnionType)]] = {k: (v, None) for k, v in init_w.items()}

    for fn, out_names in seq:
        sig = signature(fn)
        for p in sig.parameters.values():
            # cannot type check on **kwargs arguments
            if p.kind == p.VAR_KEYWORD:
                continue

            if p.kind in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL):
                raise TypeError(
                    f"function: `{fn.__name__}` has a strictly positional argument {p.name}"
                    f", which cannot be run through the pipeline mechanism of this library"
                )

            if p.name not in w_types:
                if p.default is iempty:
                    raise ValueError(
                        f"pipeline called with missing arguments, provided function {fn.__name__} expects presence of {p.name}, but it was not provided"
                    )
                continue
            if strict and p.annotation is not iempty:
                fst, snd = w_types[p.name]
                if fst is not None and not vtype_check(fst, p.annotation):
                    raise TypeError(
                        f"function: `{fn.__name__}` expected value of type "
                        f"`{p.annotation}` to be provided for argument `{p.name}`"
                        f", but value {fst} of type {type(fst)} was provided instead"
                    )
                elif snd is not None and not ttype_check(snd, p.annotation):
                    raise TypeError(
                        f"function: `{fn.__name__}` expected value of type "
                        f"`{p.annotation}` to be provided for argument `{p.name}`"
                        f", but value of type {snd} was provided instead"
                    )
        annot = sig.return_annotation
        if annot is iempty:
            if len(out_names) > 1:
                for name in out_names:
                    w_types[name] = (None, Any)  # pyright: ignore[reportArgumentType]
            else:
                w_types[out_names[0]] = (None, Any)  # pyright: ignore[reportArgumentType]
        else:
            if strict:
                if len(out_names) > 1:
                    base_annot = strip_generic(annot)
                    if not (
                        all(map(lambda x: issubclass(x, tuple), get_args(base_annot)))
                        if isinstance(base_annot, UnionType)
                        else issubclass(base_annot, tuple)
                    ):
                        raise ValueError(
                            f"pipeline function {fn.__name__} is expected to return {len(out_names)} values, "
                            f"but the provided annotation for the function shows an atomic return of {annot}"
                        )
                    expected_outs = get_args(annot)
                    if len(expected_outs) != len(out_names):
                        raise ValueError(
                            f"pipeline function {fn.__name__} is expected to return {len(out_names)} values, "
                            f"but the provided annotation indicates a total of {len(expected_outs)} values"
                        )
                    for name, tp in zip(out_names, expected_outs):
                        w_types[name] = (None, tp)
                else:
                    w_types[out_names[0]] = (None, annot)
