from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import partial
from typing import Protocol

import anndata as ad
import numpy as np
import pandas as pd

from .adapters import Pipeline, wrap_gby_fn
from .methods import bin_kmeans, reg_logit, run_pca
from .tune import Tuner


def prep_grps(
    obs: pd.DataFrame,
    grp_by: str | tuple[str],
    which_grouped: str
    | tuple[str]
    | list[str]
    | dict[str, Callable[[object, np.ndarray[tuple[int], np.dtype[np.integer]]], object]]
    | None,
    kwargs: dict,
) -> tuple[
    Mapping[object, np.ndarray[tuple[int], np.dtype[np.integer]]],
    np.ndarray[tuple[int], np.dtype[np.integer]],
    dict[str, Callable[[object, np.ndarray[tuple[int], np.dtype[np.integer]]], object]],
]:
    if obs[grp_by].isna().any(axis=None):
        raise ValueError("group by adapters cannot be used on columns containing na values.")

    grp_idx = obs.groupby(list(grp_by) if isinstance(grp_by, tuple) else grp_by, sort=True, dropna=False, observed=True).indices

    for g, idx in grp_idx.items():
        unique_per_grp = np.unique(kwargs["V"][idx])
        if len(unique_per_grp) < 2:
            raise ValueError(
                "cannot run grouped HiDDEN pipeline w/ grouping where all "
                "observations from group exist in only one category, but got "
                f"only values of {unique_per_grp} for group {g}"
            )

    fidx = lambda v, i: v[i]  # noqa: E731
    if which_grouped is None:
        which_grouped = {
            k: fidx
            for k, v in kwargs.items()
            if hasattr(v, "__getitem__")
            # short circuit check if shape attribute is present to avoid error on sparse matrices
            and ((hasattr(v, "shape") and v.shape[0] == len(obs)) or (hasattr(v, "__len__") and len(v) == len(obs)))
        }
    elif isinstance(which_grouped, str):
        which_grouped = {which_grouped: fidx}
    elif isinstance(which_grouped, list | tuple):
        which_grouped = {k: fidx for k in which_grouped}
    else:
        which_grouped["V"] = fidx

    return (grp_idx, np.argsort(np.concat(list(grp_idx.values()))), which_grouped)


def HiDDEN(
    x: ad.AnnData,
    /,
    cond_col: str,
    control_val: object,
    algo: Pipeline = Pipeline(run_pca, reg_logit, bin_kmeans, True),
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """
    runs HiDDEN on a given :py:class:`~anndata.AnnData` object.

    :param x: input :py:class:`~anndata.AnnData` object
    :param cond_col: string indicating obs column in adata representing condition value
    :param control_val: value representing the control condition in the provided condition column
    :param algo: algorithm pipeline (expected to use parameter ``V`` as original condition annotation)
    :param kwargs: additional variables to pass into pipeline
    :return: 2-tuple consisting of:

        - 1-d array of prediction outputs by model
        - binarized labels from prediction values
    """
    assert isinstance(x.obs, pd.DataFrame), "only `pandas.DataFrame` as `x.obs` currently supported"

    p_hat, labs, _ = algo(V=x.obs[cond_col].ne(control_val).to_numpy(), **kwargs)

    return (pd.Series(p_hat, index=x.obs.index), x.obs[cond_col].where(labs, control_val))


def HiDDENg(
    x: ad.AnnData,
    /,
    cond_col: str,
    control_val: object,
    group_by: str | tuple[str],
    algo: Pipeline = Pipeline(run_pca, reg_logit, bin_kmeans, True),
    which_grouped: str
    | tuple[str]
    | list[str]
    | dict[str, Callable[[object, np.ndarray[tuple[int], np.dtype[np.integer]]], object]]
    | None = None,
    grp_specific_args: Mapping[object, dict[str, object]] | None = None,
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """
    runs HiDDEN on a given :py:class:`~anndata.AnnData` object, given some set of grouping factors.

    :param x: input :py:class:`~anndata.AnnData` object
    :param cond_col: string indicating obs column in adata representing condition value
    :param control_val: value representing the control condition in the provided condition column
    :param group_by: set of column names in ``x.obs`` specifying grouping
    :param algo: algorithm pipeline (expected to use parameter ``V`` as original condition annotation)
    :param which_grouped: set of pipeline arguments which should be indexed by grouping (set to None to group all pipeline arguments which support indexing)
    :param grp_specific_args: any additional arguments that are to be provided on a group-specific basis
    :param kwargs: additional variables to pass into pipeline across every group
    :return: 2-tuple consisting of:

        - 1-d array of prediction outputs by model
        - binarized labels from prediction values
    """
    assert isinstance(x.obs, pd.DataFrame), "only `pandas.DataFrame` as `x.obs` currently supported"

    if grp_specific_args is None:
        grp_specific_args = defaultdict(dict)

    kwargs = kwargs | {"V": (x.obs[cond_col] != control_val).to_numpy()}
    grp_idx, out_ord, which_grouped = prep_grps(x.obs, group_by, which_grouped, kwargs)

    gfn = wrap_gby_fn(algo, which_grouped, grp_idx)
    p_hat, labs, _ = zip(*(gfn(grp, **(kwargs | grp_specific_args[grp])) for grp in grp_idx))

    return (
        pd.Series(np.concat(p_hat)[out_ord], index=x.obs.index),
        x.obs[cond_col].where(np.concat(labs)[out_ord], control_val),
    )


def HiDDENt[P, S](
    x: ad.AnnData,
    /,
    cond_col: str,
    control_val: object,
    tuner: Tuner[P, S],
    algo: Pipeline = Pipeline(run_pca, reg_logit, bin_kmeans, True),
    **kwargs,
) -> tuple[P, Mapping[P, tuple[pd.Series, pd.Series, S]]]:
    """
    runs HiDDEN on a given :py:class:`~anndata.AnnData` object, trying to optimize for a specific set of dimensions using provided :py:class:`~found.tune.Tuner`.

    :param x: input :py:class:`~anndata.AnnData` object
    :param cond_col: string indicating obs column in adata representing condition value
    :param control_val: value representing the control condition in the provided condition column
    :param tuner: provided tuner which attempts to optimize pipeline for a specific hyperparameter
    :param algo: algorithm pipeline (expected to use parameter ``V`` as original condition annotation)
    :param kwargs: additional variables to pass into pipeline
    :return: 2-tuple consisting of:

        - selected optimal pipeline hyperparameters
        - dictionary with keys being pipeline hyperparameters and values being a 3-tuple of:

            - 1-d array of prediction outputs by model
            - model adjusted labels
            - optional scoring value generated by tuner for specific hyperparameter configuration

    """
    assert isinstance(x.obs, pd.DataFrame), "only `pandas.DataFrame` as `x.obs` currently supported"

    best_param, outs = tuner(algo, V=x.obs[cond_col].ne(control_val).to_numpy(), **kwargs)

    return (
        best_param,
        {k: (pd.Series(Y, index=x.obs.index), x.obs[cond_col].where(W, control_val), s) for k, (Y, W, s) in outs.items()},
    )


class ByParamAccessor[G, P, S](Protocol):
    def __call__(
        self, mapping: Mapping[G, P] | None = None, default: P | None = None
    ) -> tuple[pd.Series, pd.Series, Mapping[G, S]]: ...


def HiDDENgt[P, S, G](
    x: ad.AnnData,
    /,
    cond_col: str,
    control_val: object,
    group_by: str | tuple[str],
    tuner: Tuner[P, S],
    algo: Pipeline = Pipeline(run_pca, reg_logit, bin_kmeans, True),
    which_grouped: str
    | tuple[str]
    | list[str]
    | dict[str, Callable[[object, np.ndarray[tuple[int], np.dtype[np.integer]]], object]]
    | None = None,
    grp_specific_args: Mapping[G, dict[str, object]] | None = None,
    **kwargs,
) -> tuple[
    Mapping[G, P],
    ByParamAccessor[G, P, S],
    Callable[[G], Mapping[P, tuple[pd.Series, pd.Series, S]]],
]:
    """
    runs HiDDEN on a given :py:class:`~anndata.AnnData` object, given some set of grouping factors, trying to optimize for a specific set of dimensions using provided :py:class:`~found.tune.Tuner`.

    :param x: input :py:class:`~anndata.AnnData` object
    :param cond_col: string indicating obs column in adata representing condition value
    :param control_val: value representing the control condition in the provided condition column
    :param group_by: set of column names in ``x.obs`` specifying grouping
    :param tuner: provided tuner which attempts to optimize pipeline for a specific hyperparameter
    :param algo: algorithm pipeline (expected to use parameter ``V`` as original condition annotation)
    :param which_grouped: set of pipeline arguments which should be indexed by grouping (set to None to group all pipeline arguments which support indexing)
    :param grp_specific_args: any additional arguments that are to be provided on a group-specific basis
    :param kwargs: additional variables to pass into pipeline across every group
    :return: 3-tuple consisting of:

        - mapping from each group to selected hyper-parameter for that group
        - accessor function which given a mapping of groups to hyper-parameters (and an optional default hyper-parameter for unspecified groups), returns a 3-tuple of:

            - 1-d array of prediction outputs by model, ordered by their original order within the provided
            - model adjusted labels
            - mapping of group to score value given provided configuration

        - accessor function which given a specific group, returns a HiDDENt-style output dictionary for just that group
    """
    assert isinstance(x.obs, pd.DataFrame), "only `pandas.DataFrame` as `x.obs` currently supported"

    if grp_specific_args is None:
        grp_specific_args = defaultdict(dict)

    kwargs = kwargs | {"V": (x.obs[cond_col] != control_val).to_numpy()}
    grp_idx, out_ord, which_grouped = prep_grps(x.obs, group_by, which_grouped, kwargs)

    gfn = wrap_gby_fn(partial(tuner, algo), which_grouped, grp_idx)
    best_params, outs = zip(
        *(
            gfn(
                grp,
                **(
                    kwargs  # fmt: skip
                    | grp_specific_args[grp]  # ty:ignore[invalid-argument-type]
                    # ignore NECESSITY - grp should always be hashable? could use more validation
                ),
            )
            for grp in grp_idx
        )
    )
    best_params = {g: h for g, h in zip(grp_idx.keys(), best_params, strict=True)}
    outs = {g: o for g, o in zip(grp_idx.keys(), outs, strict=True)}

    def acc_by_param(
        mapping: Mapping[G, P] | None = None, default: P | None = None
    ) -> tuple[pd.Series, pd.Series, Mapping[G, S]]:

        if mapping is None:
            mapping = dict()
        if default is None:
            if set(mapping.keys()) != set(grp_idx.keys()):
                raise ValueError(
                    f"provided mapping {mapping} does not span all present groups {grp_idx.keys()} and no default was provided"
                )

        def get(g: G) -> P:
            out = mapping[g] if g in mapping else default
            assert out is not None
            return out

        return (  # ty:ignore[invalid-return-type]
            # ignore NECESSITY - ???
            pd.Series(
                np.concat(
                    [
                        outs[g][
                            get(g)  # ty:ignore[invalid-argument-type]
                        ][0]
                        for g in grp_idx.keys()
                    ]
                )[out_ord],
                index=x.obs.index,
            ),
            x.obs[cond_col].where(  # ty:ignore[unresolved-attribute]
                # ignore NECESSITY - typechecker does not catch isinstance check at top of function
                #                    validatiing that x.obs should always be pd.DataFrame
                np.concat(
                    [
                        outs[g][
                            get(g)  # ty:ignore[invalid-argument-type]
                        ][1]
                        for g in grp_idx.keys()
                    ]
                )[out_ord],
                control_val,
            ),
            {
                g: outs[g][
                    get(g)  # ty:ignore[invalid-argument-type]
                ][2]
                for g in grp_idx.keys()
            },
        )
        # ignore NECESSITY - g should be of type G? could use more validation

    def acc_by_grp(grp: G) -> Mapping[P, tuple[pd.Series, pd.Series, S]]:
        return {
            k: (
                pd.Series(Y, index=x.obs.index[grp_idx[grp]]),
                x[grp_idx[grp]]
                .obs[cond_col]
                .where(  # ty:ignore[unresolved-attribute]
                    # ignore NECESSITY - typechecker does not catch isinstance check at top of function
                    #                    validatiing that x.obs should always be pd.DataFrame
                    W,
                    control_val,
                ),
                s,
            )
            for k, (Y, W, s) in outs[grp].items()
        }

    return (
        best_params,
        acc_by_param,
        acc_by_grp,
    )
