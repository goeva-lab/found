from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import altair as alt
import anndata as ad
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray
from scipy import sparse as sp
from scipy.stats import gaussian_kde

alt.data_transformers.enable("vegafusion")


def mk_range(d, n: int) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return (np.arange(n) / n) * (np.max(d) - np.min(d)) + np.min(d)


def kde(d, n: int) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    if d.size <= 1:
        return np.repeat(1.0, n)
    return gaussian_kde(d)(mk_range(d, n))


class PlotAdata:
    # @TODO: add type specifications for fields, do verification

    def __init__(self, adata: ad.AnnData | None, layer: str | None = None):
        if adata is not None:
            if layer is not None:
                assert layer in adata.layers
                self.__mtx = adata.layers[layer]
            else:
                assert adata.X is not None
                self.__mtx = adata.X

            self.__idx = adata.var.index
            self.__meta = adata.obs
            self.__meta_m = adata.obsm
            self.__meta_p = adata.obsp
            self.__cache = {}

    def __ret_cache(self, key: str, val) -> ExtensionArray:
        val = pd.Series(val).array
        self.__cache[key] = val
        return val

    def get_data(self, key: str) -> ExtensionArray:
        if self.__mtx is None:
            raise ValueError(
                "using get_data cannot be done when plotting object has not been initialized with an anndata object"
            )

        if key in self.__cache:
            return self.__cache[key]

        if key in self.__idx:
            o = self.__mtx[:, self.__idx.get_loc(key)]  # ty:ignore[not-subscriptable]
            # ignore NECESSITY - adata.X type specifications are so broad
            if isinstance(o, sp.sparray | sp.spmatrix):
                o = o.todense()  # ty:ignore[unresolved-attribute]
                # ignore NECESSITY - ???
                o = np.asarray(o).reshape(-1)
            return self.__ret_cache(key, o)

        if key in self.__meta.columns:
            o = self.__meta[key]
            return self.__ret_cache(key, o)

        split = key.split(".", maxsplit=1)
        if len(split) == 2:
            key_b, idx = split
            idx = int(idx)
            if key_b in self.__meta_m:
                o = self.__meta_m[key_b][:, idx]
                return self.__ret_cache(key, o)

        raise ValueError(f"key {key} invalid")

    def encode(self, *encode_args: alt.SchemaBase) -> alt.Chart:
        conf = {k._kwds["shorthand"].replace(".", "_"): k._kwds["shorthand"] for k in encode_args}
        chk = set(map(lambda k: k._kwds["shorthand"], encode_args))
        assert len(chk) == len(conf), f"configuration schema ambiguous, please rename columns: {chk - set(conf.values())}"

        df = {k: self.get_data(v) for k, v in conf.items()}
        schemas = []
        for schema in encode_args:
            schema = schema.title(schema._kwds["shorthand"])
            schema = type(schema)(**(schema._kwds | {"shorthand": schema._kwds["shorthand"].replace(".", "_")}))
            if schema._kwds["sort"] is alt.Undefined and df[schema._kwds["shorthand"]].dtype.kind == "O":
                try:
                    to_num = pd.to_numeric(df[schema._kwds["shorthand"]])
                except (ValueError, TypeError):
                    pass
                else:
                    schema = schema.sort(np.unique(to_num).astype(type(df[schema._kwds["shorthand"]][0])).tolist())

            schemas.append(schema)

        return alt.Chart(pd.DataFrame(df)).encode(*schemas)

    def point(self, *encode_args: alt.SchemaBase, **mark_args) -> alt.Chart:
        return self.encode(*encode_args).mark_point(**({"filled": True, "opacity": 1} | mark_args))

    def line(self, *encode_args: alt.SchemaBase, **mark_args) -> alt.Chart:
        return self.encode(*encode_args).mark_line(**mark_args)

    def vln(
        self,
        continuous: str | pd.Series,
        discrete: str | pd.Series | None = None,
        split: str | pd.Series | None = None,
        rescale_by: Literal["area", "width", "n"] = "n",
        vertical: bool = True,
        n: int = 100,
    ) -> alt.Chart:
        data = pd.DataFrame(
            {"continuous": self.get_data(continuous) if isinstance(continuous, str) else continuous.array}
            | (
                {}
                if discrete is None
                else {"discrete": self.get_data(discrete) if isinstance(discrete, str) else discrete.array}
            )
            | ({} if split is None else {"split": self.get_data(split) if isinstance(split, str) else split.array})
        )

        if split is not None:
            assert len(data["split"].unique()) == 2

        gby = None
        match discrete is None, split is None:
            case False, False:
                gby = ["discrete", "split"]
            case True, False:
                gby = "split"
            case False, True:
                gby = "discrete"

        if gby is None:
            dens_data = pd.DataFrame({"density": kde(data["continuous"], n), "val": mk_range(data["continuous"], n)})
        else:
            dens_data = (data if gby is None else data.groupby(gby, observed=True))["continuous"].agg(
                density=lambda d: list(kde(d, n)), val=lambda d: list(mk_range(d, n))
            )

            if split is not None and rescale_by == "n":
                counts = data.groupby(gby, observed=True)["continuous"].count()
                if discrete is not None:
                    counts = counts.groupby("discrete", observed=True)

                dens_data = dens_data.join(counts.transform(lambda e: e / e.sum()))

            assert isinstance(dens_data, pd.DataFrame)  # for type checker, should always be true

            dens_data = dens_data.explode(["density", "val"]).reset_index()

            if split is not None and rescale_by == "n":
                dens_data["density"] = dens_data["density"] * dens_data["continuous"]

        if discrete is None:
            dens_data["density"] = (dens_data["density"] - dens_data["density"].min()) / (
                dens_data["density"].max() - dens_data["density"].min()
            )
        else:
            dens_data["density"] = dens_data.groupby(
                ["discrete", "split"] if rescale_by == "width" else "discrete", observed=True
            )["density"].transform(lambda d: (d - np.min(d)) / (np.max(d) - np.min(d)))

        if split is None:
            dens_data["density_refl"] = -dens_data["density"]
        else:
            dens_data["density"] = dens_data["density"].mask(
                dens_data["split"] == dens_data["split"].unique()[0], -dens_data["density"]
            )
            dens_data["density_refl"] = 0

        if vertical:
            dens_ax, refl_ax, val_ax, facet_ax = alt.X, alt.X2, alt.Y, alt.Column
            mdict = {"orient": "horizontal"}
            # annotation needed to avoid narrowing by typecheker
            hdict: dict = {"orient": "bottom", "labelAnchor": "middle", "labelPadding": 2}
        else:
            dens_ax, refl_ax, val_ax, facet_ax = alt.Y, alt.Y2, alt.X, alt.Row
            mdict = {"orient": "vertical"}
            # annotation needed to avoid narrowing by typecheker
            hdict: dict = {"labelPadding": 2, "labelAngle": 0, "labelAlign": "left"}

        chart = (
            alt.Chart(dens_data.sort_values("val"), view=alt.ViewConfig(stroke=None))
            .encode(
                dens_ax("density:Q")
                .impute(None)
                .title(None)
                .axis(labels=False, grid=False, values=[] if discrete is None else [0])
                .scale(domain=[-1.1, 1.1]),
                val_ax("val:Q").title(
                    continuous
                    if isinstance(continuous, str)
                    else (str(continuous.name) if continuous.name is not None else None)
                ),
                refl_ax("density_refl:Q"),
            )
            .mark_area(**mdict)
        )

        if discrete is not None:
            facet_enc = facet_ax("discrete:N").spacing(0).title(None).header(**hdict)
            if hasattr(data["discrete"], "cat"):
                facet_enc = facet_enc.sort(data["discrete"].cat.categories)
            chart = chart.encode(facet_enc)

        if split is not None:
            col_enc = alt.Color("split:N").title(
                split if isinstance(split, str) else (str(split.name) if split.name is not None else None)
            )
            if hasattr(data["split"], "cat"):
                col_enc = col_enc.sort(data["split"].cat.categories)
            chart = chart.encode(col_enc)

        return chart


@dataclass(frozen=True)
class PlotHiDDENOutput:
    """
    class which when provided with pipeline outputs (see fields),
    can be used to create various diagnostic plots.

    can be indexed to plot only a subset of the data.

    :param adata: object containing input data
    :param p_hat: HiDDEN-generated p_hat values (regression step outputs)
    :param labs: HiDDEN-adjusted condition labels (binarization step outputs)
    :param layer: name of layer to use when accessing layer-specific data from ``self.adata``, ``X`` slot is used if set to ``None``
    """

    adata: ad.AnnData
    p_hat: pd.Series
    labs: pd.Series
    layer: str | None = field(kw_only=True, default=None)

    __pl: PlotAdata = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            # make sure to mangle private __pl field, see scheme defined here: https://peps.python.org/pep-0008/#method-names-and-instance-variables
            f"_{type(self).__name__}__pl",
            PlotAdata(self.adata, layer="counts" if ((self.adata.X is None) and (self.layer is None)) else self.layer),
        )

    def __getitem__(self, idx) -> Self:
        if isinstance(idx, Callable):
            idx = idx(self.adata)

        return type(self)(
            self.adata[idx],
            self.p_hat[idx],
            self.labs[idx],
            layer=self.layer,
        )

    def reg_vln(
        self,
        group_by: str | pd.Series | None = None,
        split_mode: Literal[False, "area", "width", "n"] = "n",
        vertical: bool = True,
        n: int = 100,
    ) -> alt.Chart:
        """
        method used to generate violin plots of p_hat values

        :param group_by: key in self.adata to group violin plot by
        :param split_mode: if the violin plot should be split by the refined HiDDEN labels, and if yes, how should the splits be scaled
            (options: ``"area"`` - meaning they have equivalent area, ``"n"`` - scaled by proportion of observations, ``"width"`` - min/max scaling so that they have the same width, ``False`` - no splitting)
        :param vertical: should the plot be vertical (like a violin plot) or horizontal (e.g. like a density plot)
        :param n: number of bins for density estimation
        """
        return (
            self.__pl.vln(
                pd.Series(self.p_hat, name="HiDDEN p_hat"),
                group_by,
                pd.Series(self.labs, name="HiDDEN labels"),
                rescale_by=split_mode,
                vertical=vertical,
                n=n,
            )
            if split_mode is not False
            else self.__pl.vln(
                pd.Series(self.p_hat, name="HiDDEN p_hat"),
                group_by,
                None,
                vertical=vertical,
                n=n,
            )
        )

    def bin_bar(
        self,
        orig_labs: str | pd.Series,
        ctrl_val: object,
        group_by: str | pd.Series | None = None,
        vertical: bool = True,
        scale: bool = True,
    ) -> alt.Chart:
        """
        method used to generate stacked bar plots showing levels of case, control, and HiDDEN-adjusted labels

        :param orig_labs: key in ``self.adata`` for original condition labels
        :param ctrl_val: value corresponding to control in condition labels
        :param group_by: optional key in ``self.adata`` to group calculation by
        :param vertical: should the plot be vertical (i.e. metric on Y axis, groups on X axis) or horizontal (i.e. metric on X axis, groups on Y axis)
        :param scale: should columns height be scaled by group size (i.e. turns into representation of group fractions instead raw numbers)
        """
        if isinstance(orig_labs, str):
            orig_labs = pd.Series(self.__pl.get_data(orig_labs), name=orig_labs)

        ctrl_str, unaff_str, aff_str = "control", "case: HiDDEN - unaffected", "case: HiDDEN - affected"
        df = pd.DataFrame(
            {
                "label": pd.Series(np.where(np.asarray(self.labs == ctrl_val), unaff_str, aff_str)).mask(
                    orig_labs == ctrl_val, ctrl_str
                )
            }
        )

        if group_by is not None:
            if isinstance(group_by, str):
                group_by = pd.Series(self.__pl.get_data(group_by), name=group_by)
            assert isinstance(group_by.name, str)
            df[group_by.name] = group_by

            df = df.groupby([group_by.name, "label"], observed=True)
        else:
            df = df.groupby("label", observed=True)

        col = "proportion" if scale else "count"

        df = df.agg("size").rename(col)

        if scale:
            if group_by is not None:
                df = df.groupby(group_by.name, observed=True)
            df = df.transform(lambda x: x / x.sum())

        c = alt.Chart(df.to_frame().reset_index()).encode(
            (alt.Y if vertical else alt.X)(f"{col}:Q"),
            alt.Color("label:N").sort([ctrl_str, unaff_str, aff_str]),
            alt.Order("color_label_sort_index:Q").sort("descending" if vertical else "ascending"),
        )

        if group_by is not None:
            c = c.encode((alt.X if vertical else alt.Y)(group_by.name))  # ty:ignore[invalid-argument-type]
            # ignore NECESSITY - typechecker does not catch group_by.name isinstance check verifying it to be a string

        return c.mark_bar()


@dataclass(frozen=True)
class PlotTunerOutput:
    """
    class which when provided with tuner outputs (see fields),
    can be used to create various diagnostic plots.

    can be indexed to return PlotHiDDENOutput objects for diagnostics on specific hyperparameters.

    :param adata: object containing input data
    :param sel: HiDDENt-returned selected hyperparameter configuration
    :param outs: HiDDENt-returned output dictionary
    :param layer: name of layer to use when accessing layer-specific data from ``self.adata``, ``X`` slot is used if set to ``None``
    """

    adata: ad.AnnData
    sel: object
    outs: Mapping[Any, tuple[pd.Series, pd.Series, Any]]
    layer: str | None = field(kw_only=True, default=None)

    def __getitem__(self, k: Any) -> PlotHiDDENOutput:
        if k not in self.outs:
            raise ValueError(
                f"provided key must be one of evaluated hyperparameters, but {k} is not in {set(self.outs.keys())}"
            )
        return PlotHiDDENOutput(self.adata, self.outs[k][0], self.outs[k][1])

    def score_line(self) -> alt.Chart | alt.LayerChart:
        """
        method used to generate a line plot of scores for tested hyperparameters
        """
        params = sorted(self.outs.keys())
        c = alt.Chart(
            pd.DataFrame(
                {
                    "hyperparameter": params,
                    "score": [self.outs[k][2] for k in params],
                    "selected": [p == self.sel for p in params],
                }
            )
        ).encode(alt.X("hyperparameter"), alt.Y("score"))

        return c.mark_line() + c.encode(alt.Color("selected")).mark_point(filled=True, opacity=1)
