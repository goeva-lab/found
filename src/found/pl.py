from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

import altair as alt
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import gaussian_kde

from .types import NumArr

alt.data_transformers.enable("vegafusion")


def mk_range(d, n: int) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    return (np.arange(n) / n) * (np.max(d) - np.min(d)) + np.min(d)


def kde(d, n: int) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    if d.size <= 1:
        return np.repeat(1.0, n)  # pyright: ignore
    return gaussian_kde(d)(mk_range(d, n))


class PlotAdata:
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

    def __ret_cache(self, key: str, val) -> np.ndarray:
        val = np.asarray(val)
        self.__cache[key] = val
        return val

    def get_data(self, key: str) -> Any:
        if self.__mtx is None:
            raise ValueError(
                "using get_data cannot be done when plotting object has not been initialized with an anndata object"
            )

        if key in self.__cache:
            return self.__cache[key]

        if key in self.__idx:
            o = self.__mtx[:, self.__idx.get_loc(key)]
            if isinstance(o, sp.sparray | sp.spmatrix):
                o = o.todense()  # pyright: ignore
                o = np.array(o).reshape(-1)
            return self.__ret_cache(key, o)

        if key in self.__meta.columns:
            o = self.__meta[key].values
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
                    schema = schema.sort(
                        np.unique(
                            to_num,  # pyright: ignore
                        )
                        .astype(type(df[schema._kwds["shorthand"]][0]))
                        .tolist()
                    )

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
        rescale_by: Literal[None, "width", "n"] = "n",
        vertical: bool = True,
        n: int = 100,
    ) -> alt.Chart:
        data = pd.DataFrame(
            {"continuous": self.get_data(continuous) if isinstance(continuous, str) else continuous.values}
            | (
                {}
                if discrete is None
                else {"discrete": self.get_data(discrete) if isinstance(discrete, str) else discrete.values}
            )
            | ({} if split is None else {"split": self.get_data(split) if isinstance(split, str) else split.values})
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
            mdict: dict = {"orient": "horizontal"}
            hdict = {"orient": "bottom", "labelAnchor": "middle", "labelPadding": 2}
        else:
            dens_ax, refl_ax, val_ax, facet_ax = alt.Y, alt.Y2, alt.X, alt.Row
            mdict: dict = {"orient": "vertical"}
            hdict = {"labelPadding": 2, "labelAngle": 0, "labelAlign": "left"}

        chart = (
            alt.Chart(dens_data.sort_values("val"), view=alt.ViewConfig(stroke=None))
            .encode(
                dens_ax("density")
                .impute(None)
                .title(None)
                .axis(labels=False, grid=False, values=[] if discrete is None else [0])
                .scale(domain=[-1.1, 1.1]),
                val_ax("val").title(
                    continuous
                    if isinstance(continuous, str)
                    else (str(continuous.name) if continuous.name is not None else None)
                ),
                refl_ax("density_refl"),
            )
            .mark_area(**mdict)
        )

        if discrete is not None:
            chart = chart.encode(facet_ax("discrete").spacing(0).title(None).header(**hdict))

        if split is not None:
            chart = chart.encode(
                alt.Color("split").title(
                    split if isinstance(split, str) else (str(split.name) if split.name is not None else None)
                )
            )

        return chart


@dataclass(frozen=True)
class PlotHiDDENOutput:
    """
    class which when provided with pipelien outputs (see fields),
    can be used to create various diagnostic plots.

    can be indexed to plot only a subset of the data.

    :param adata: anndata object containing input data
    :param phat: HiDDEN-generated p hat values
    :param labs: HiDDEN-adjusted condition labels
    """

    adata: ad.AnnData
    phat: NumArr
    labs: np.ndarray[tuple[int], Any]

    def __getitem__(self, idx) -> Self:
        if isinstance(idx, Callable):
            idx = idx(self.adata)

        return type(self)(
            self.adata[idx],
            self.phat[idx],  # pyright: ignore
            self.labs[idx],  # pyright: ignore
        )

    def phat_vln(
        self,
        orig_labs: str | pd.Series | None = None,
        group_by: str | pd.Series | None = None,
        rescale_by: Literal[None, "width", "n"] = "n",
        vertical: bool = True,
        n: int = 100,
    ) -> alt.Chart:
        """
        method used to generate violin plots of p hat values

        :param orig_labs: key in self.adata for original condition labels (will be used to split violin plot on if HiDDEN changed the label)
        :param group_by: key in self.adata to group violin plot by
        :param rescale_by_n: if the violin plot is split, how should the two sides be scaled?
            (options: None - meaning they have equivalent area, "n" - scaled by proportion of observations, "width" - min/max scaling so that they have the same width)
        :param vertical: should the plot be vertical (like a violin plot) or horizontal (e.g. like a density plot)
        :param n: number of bins for density estimation
        """
        pl = PlotAdata(self.adata)
        if isinstance(orig_labs, str):
            orig_labs = pd.Series(pl.get_data(orig_labs), name=orig_labs)
        return pl.vln(
            pd.Series(self.phat, name="HiDDEN_phat"),
            group_by,
            pd.Series(self.labs != orig_labs, name="HiDDEN_changed") if orig_labs is not None else None,
            rescale_by=rescale_by,
            vertical=vertical,
            n=n,
        )

    def labs_pct(
        self, orig_labs: str | pd.Series, ctrl_val: Any, group_by: str | pd.Series | None = None, vertical: bool = True
    ) -> alt.LayerChart:
        """
        method used to generate point plots of percentages of case vs control cells in original vs HiDDEN-adjusted labels

        :param orig_labs: key in self.adata for original condition labels
        :param ctrl_val: value corresponding to control in condition labels
        :param group_by: key in self.adata to group calculation by
        :param vertical: should the plot be vertical (i.e. metric on Y axis, groups on X axis) or horizontal (i.e. metric on X axis, groups on Y axis)
        """
        pl = PlotAdata(self.adata)

        if isinstance(orig_labs, str):
            orig_labs = pd.Series(pl.get_data(orig_labs), name=orig_labs)

        if group_by is None:
            c = alt.Chart(
                pd.DataFrame(
                    {
                        "pct_case": [pd.Series(orig_labs != ctrl_val).mean(), pd.Series(self.labs != ctrl_val).mean()],
                        "source": ["original", "HiDDEN"],
                    }
                )
            )
        else:
            if isinstance(group_by, str):
                group_by = pd.Series(pl.get_data(group_by), name=group_by)
            assert isinstance(group_by.name, str)

            c = alt.Chart(
                pd.concat(
                    [
                        pd.Series(orig_labs != ctrl_val)
                        .groupby(group_by)
                        .agg("mean")
                        .to_frame(name="pct_case")
                        .assign(source="original"),
                        pd.Series(self.labs != ctrl_val)
                        .groupby(group_by)
                        .agg("mean")
                        .to_frame(name="pct_case")
                        .assign(source="HiDDEN"),
                    ]
                ).reset_index(names=group_by.name)
            ).encode(
                (alt.X if vertical else alt.Y)(group_by.name),
                alt.Detail(group_by.name),
            )

        c = c.encode((alt.Y if vertical else alt.X)("pct_case").scale(domain=[0, 1]))

        return c.mark_line(
            strokeDash=(4, 4),
        ) + c.mark_point(
            filled=True,
            opacity=0.8,
        ).encode(
            alt.Color("source"),
            alt.Shape("source"),
        )


@dataclass(frozen=True)
class PlotTunerOutput:
    """
    class which when provided with tuner outputs (see fields),
    can be used to create various diagnostic plots.

    can be indexed to return PlotHiDDENOutput objects for diagnostics on specific hyperparameters.

    :param adata: anndata object containing input data
    :param outs: HiDDENt-returned output dictionary
    :param labs: selected optimal hyperparameter
    """

    adata: ad.AnnData
    outs: Mapping
    sel: Any

    def __getitem__(self, k) -> PlotHiDDENOutput:
        if k not in self.outs:
            raise ValueError(
                f"provided key must be one of evaluated hyperparameters, but {k} is not in {set(self.outs.keys())}"
            )
        return PlotHiDDENOutput(self.adata, self.outs[k][0], self.outs[k][1])

    def plot_scores(self) -> alt.Chart:
        """
        method used to generate a line plot of scores for tested hyperparameters
        """
        params = sorted(self.outs.keys())
        return (
            alt.Chart(
                pd.DataFrame(
                    {
                        "hyperparameter": params,
                        "score": [self.outs[k][2] for k in params],
                        "selected": [p == self.sel for p in params],
                    }
                )
            )
            .encode(alt.X("hyperparameter"), alt.Y("score"), alt.Shape("selected"))
            .mark_line(point=True)
        )
