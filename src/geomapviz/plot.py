"""Small native renderers for prepared geographic summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

from .aggregator import _add_column
from .shapefiles import _boundaries
from .utils import check_list_of_str

if TYPE_CHECKING:
    import geopandas as gpd
    import holoviews as hv
    from matplotlib.figure import Figure

__all__ = ["plot_geography", "PlotOptions"]


@dataclass
class PlotOptions:
    """Rendering only: size in inches (100 pixels/inch interactively).

    ``autobin`` pools finite values within each quantity and uses at most
    ``n_bins`` Fisher-Jenks classes. Differences use symmetric equal intervals
    instead, so zero stays at the center. Neither path treats missing as zero.
    """

    figsize: tuple[float, float] = (12, 8)
    ncols: int = 2
    cmap: str = "viridis"
    facecolor: str = "white"
    alpha: float = 1.0
    autobin: bool = False
    n_bins: int = 7
    interactive: bool = False

    def __post_init__(self):
        for name in ("ncols", "n_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and between 0 and 1")
        if len(self.figsize) != 2 or not all(
            np.isfinite(size) and size > 0 for size in self.figsize
        ):
            raise ValueError("figsize must contain two finite positive sizes")
        for name in ("interactive", "autobin"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        mpl.colors.to_rgba(self.facecolor)
        mpl.colormaps[self.cmap]


def _quantities(mapped):
    """Use actual derived names, including those changed by collisions."""
    # ponytail: untagged metrics share units; add explicit groups if mixed-unit
    # summaries without metadata become a supported workflow.
    quantities = {name: "total" for name in mapped.attrs.get("totals", {}).values()}
    for comparison in mapped.attrs.get("comparisons", {}).values():
        quantities.update({name: kind for kind, name in comparison.items()})
    for kind, name in mapped.attrs.get("support", {}).items():
        quantities[name] = f"support_{kind}"
    return quantities


def _scales(mapped, metrics, quantities, options):
    scales = {}
    for quantity in dict.fromkeys(quantities.get(name, "metric") for name in metrics):
        columns = [
            name for name in metrics if quantities.get(name, "metric") == quantity
        ]
        values = mapped[columns].to_numpy(dtype=float, na_value=np.nan).ravel()
        finite = values[np.isfinite(values)]
        vmin, vmax = (
            (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
        )
        if quantity == "difference":
            vmax = max(abs(vmin), abs(vmax), 0.5)
            vmin = -vmax
        elif vmin == vmax:
            padding = max(abs(vmin) * 0.05, 0.5)
            vmin, vmax = vmin - padding, vmax + padding
        with np.errstate(over="ignore", invalid="ignore"):
            if not np.isfinite([vmin, vmax, np.float64(vmax) - vmin]).all():
                raise ValueError(f"Values in {columns} exceed a finite plotting scale")
        bins = None
        if options.autobin and finite.size:
            k = min(options.n_bins, np.unique(finite).size)
            if quantity == "difference":
                k = options.n_bins - (options.n_bins % 2 == 0)
                bins = np.linspace(vmin, vmax, k + 1)[1:]
            else:
                from mapclassify import FisherJenks

                if k == 1:
                    bins = np.array([finite.max()])
                else:
                    # Fisher-Jenks variance loses precision on tightly spaced
                    # values. Classify on [0, 1], then take original data edges.
                    ordered = np.sort(finite)
                    scaled = (ordered - ordered[0]) / (ordered[-1] - ordered[0])
                    bins = ordered[
                        np.searchsorted(scaled, FisherJenks(scaled, k=k).bins)
                    ]
        for name in columns:
            scales[name] = (vmin, vmax, bins)
    return scales


def _colors(mapped, metric, scale, cmap):
    """Share exact class membership and palette across both backends."""
    vmin, vmax, bins = scale
    if bins is None:
        return (
            mapped[metric].to_numpy(dtype=float, na_value=np.nan),
            Normalize(vmin, vmax),
            mpl.colormaps[cmap],
            None,
        )
    values = mapped[metric].to_numpy(dtype=float, na_value=np.nan)
    classes = np.where(
        np.isnan(values), np.nan, np.searchsorted(bins, values, side="left")
    )
    positions = np.linspace(0, 1, len(bins)) if len(bins) > 1 else [0.5]
    palette = ListedColormap(mpl.colormaps[cmap](positions))
    norm = BoundaryNorm(np.arange(len(bins) + 1) - 0.5, len(bins))
    lower = [vmin] + bins[:-1].tolist()
    edges = lower + [bins[-1]]
    for precision in range(3, 18):
        formatted = [f"{edge:.{precision}g}" for edge in edges]
        if len(set(formatted)) == len(set(edges)):
            break
    labels = [f"{lo} < x ≤ {hi}" for lo, hi in zip(formatted, formatted[1:])]
    labels[0] = labels[0].replace(" < ", " ≤ ", 1)
    return classes, norm, palette, labels


def _text_color(facecolor):
    rgb = np.asarray(mpl.colors.to_rgb(facecolor))
    return "black" if rgb @ [0.299, 0.587, 0.114] > 0.5 else "white"


def plot_geography(
    mapped: gpd.GeoDataFrame,
    metrics: list[str],
    *,
    geoid: str,
    include_support: bool = False,
    options: PlotOptions | None = None,
) -> Figure | hv.Layout:
    """Render prepared summaries without aggregating or changing caller data.

    Returns a Matplotlib Figure or, with ``interactive=True``, a HoloViews
    Layout of GeoViews Polygons. Export using ``figure.savefig`` or ``hv.save``.
    A single metric is a single map. ``ncols`` also supports incomplete grids.

    Original metrics share a scale: select metrics measuring the same quantity.
    M2 metadata separates totals, differences (centered on zero), ratios, counts
    and weight/exposure. Preserve ``attrs`` when selecting or exporting data.
    Without metadata, all selected metrics are assumed to be comparable; plot
    different quantities in separate calls. ``include_support`` appends both
    support columns from that metadata, each on its own scale. Interactive hover
    always includes the ID, raw metric and available support columns.

    Missing/undefined values are grey with an outline, and hatched statically.
    Entirely missing panels show that state without a numerical colorbar.
    No tiles, external downloads, global styles or statistical intervals.
    """
    _boundaries(mapped, geoid)
    if not mapped.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Map geometries must be polygons or multipolygons")
    check_list_of_str(metrics, "metrics")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a non-empty list of unique column names")
    if not isinstance(include_support, bool):
        raise TypeError("include_support must be a boolean")
    options = PlotOptions() if options is None else options
    if not isinstance(options, PlotOptions):
        raise TypeError("options must be PlotOptions")
    # Validate again because dataclass fields can be edited after construction.
    options.__post_init__()
    support = list(dict.fromkeys(mapped.attrs.get("support", {}).values()))
    if include_support and not support:
        raise ValueError("include_support requires summary support metadata")
    metrics = list(dict.fromkeys(metrics + (support if include_support else [])))
    for name in list(dict.fromkeys(metrics + support)):
        if name not in mapped.columns:
            raise ValueError(f"Missing metric/support column {name!r}")
        if (
            name == geoid
            or not pd.api.types.is_numeric_dtype(mapped[name])
            or pd.api.types.is_complex_dtype(mapped[name])
        ):
            raise TypeError(
                f"Column {name!r} must contain real numeric values and not be the ID"
            )
        if np.isinf(mapped[name].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Column {name!r} contains infinite values")
    quantities = _quantities(mapped)
    scales = _scales(mapped, metrics, quantities, options)
    if options.interactive:
        return _interactive(
            mapped, metrics, geoid, support, scales, quantities, options
        )
    return _static(mapped, metrics, scales, quantities, options)


def _static(mapped, metrics, scales, quantities, options):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ncols = min(options.ncols, len(metrics))
    nrows = (len(metrics) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=options.figsize,
        facecolor=options.facecolor,
        layout="constrained",
    )
    text_color = _text_color(options.facecolor)
    for metric, axis in zip(metrics, axes.flat):
        cmap = "RdBu_r" if quantities.get(metric) == "difference" else options.cmap
        values, norm, palette, labels = _colors(mapped, metric, scales[metric], cmap)
        axis.set_facecolor(options.facecolor)
        missing_style = {
            "color": "#bdbdbd",
            "edgecolor": text_color,
            "hatch": "///",
            "label": "Missing / undefined",
        }
        if np.isnan(values).all():
            mapped.plot(
                ax=axis,
                alpha=options.alpha,
                **{k: v for k, v in missing_style.items() if k != "label"},
            )
        else:
            mapped.plot(
                column=values,
                ax=axis,
                norm=norm,
                cmap=palette,
                alpha=options.alpha,
                linewidth=0.3,
                edgecolor=text_color,
                missing_kwds=missing_style,
            )
            bar = figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=palette), ax=axis
            )
            bar.ax.tick_params(colors=text_color)
            bar.outline.set_edgecolor(text_color)
            if labels is not None:
                bar.set_ticks(range(len(labels)), labels=labels)
        if np.isnan(values).any():
            axis.legend(
                handles=[
                    Patch(
                        facecolor="#bdbdbd",
                        edgecolor=text_color,
                        hatch="///",
                        label="Missing / undefined",
                    )
                ],
                labelcolor=text_color,
                facecolor=options.facecolor,
                loc="lower left",
            )
        axis.set_title(metric, color=text_color)
        axis.set_axis_off()
    for axis in list(axes.flat)[len(metrics) :]:
        figure.delaxes(axis)
    return figure


def _interactive(mapped, metrics, geoid, support, scales, quantities, options):
    import cartopy.crs as ccrs
    import geoviews as gv
    import holoviews as hv
    from bokeh.models import FixedTicker, HoverTool

    # Initialize only on the explicitly requested interactive path.
    hv.extension("bokeh", logo=False)
    projected = mapped.to_crs(epsg=3857)
    text_color = _text_color(options.facecolor)
    ncols = min(options.ncols, len(metrics))
    nrows = (len(metrics) + ncols - 1) // ncols

    def style(plot, element):
        plot.state.border_fill_color = options.facecolor
        plot.state.title.text_color = text_color

    panels = []
    for metric in metrics:
        cmap = "RdBu_r" if quantities.get(metric) == "difference" else options.cmap
        values, norm, palette, labels = _colors(mapped, metric, scales[metric], cmap)
        frame = projected.copy()
        color = metric
        if labels is not None:
            color = _add_column(frame, "_color_class", values)
        names = list(dict.fromkeys([metric, geoid] + support + [color]))
        # Native aliases avoid collisions with coordinate/color fields and
        # HoloViews' sanitization of caller names; labels and raw data survive.
        aliases = {
            name: _add_column(frame, f"geomapviz_value_{i}", frame[name])
            for i, name in enumerate(names)
        }
        dimensions = [hv.Dimension((aliases[name], name)) for name in names]
        hover = HoverTool(
            tooltips=[
                (name, "@{" + aliases[name] + "}")
                for name in dict.fromkeys([geoid, metric] + support)
            ]
        )
        colorbar_opts = {
            "major_label_text_color": text_color,
            "title_text_color": text_color,
        }
        if labels is not None:
            colorbar_opts.update(
                ticker=FixedTicker(ticks=list(range(len(labels)))),
                major_label_overrides=dict(enumerate(labels)),
            )
        panel = gv.Polygons(frame, vdims=dimensions, crs=ccrs.GOOGLE_MERCATOR).opts(
            color=hv.dim(hv.Dimension((aliases[color], color))),
            cmap=[mpl.colors.to_hex(c) for c in palette(np.linspace(0, 1, palette.N))],
            clim=(norm.vmin, norm.vmax),
            colorbar=not np.isnan(values).all(),
            tools=[hover],
            alpha=options.alpha,
            line_color=text_color,
            line_width=0.3,
            clipping_colors={"NaN": "#bdbdbd"},
            title=metric
            + (" (missing / undefined in grey)" if np.isnan(values).any() else ""),
            width=max(1, round(options.figsize[0] * 100 / ncols)),
            height=max(1, round(options.figsize[1] * 100 / nrows)),
            xaxis=None,
            yaxis=None,
            bgcolor=options.facecolor,
            show_grid=False,
            colorbar_opts=colorbar_opts,
            shared_axes=False,
            hooks=[style],
        )
        panels.append(panel)
    return hv.Layout(panels).cols(ncols)
