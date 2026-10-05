"""Rendering checks inspect numbers, layers and native backend configuration."""

import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from geopandas.testing import assert_geodataframe_equal  # noqa: E402

from geomapviz import aggregate_means, aggregate_rates, prepare_geography  # noqa: E402
from geomapviz.plot import PlotOptions, plot_geography  # noqa: E402


def sample():
    from examples.native_comparison import sample as native_sample

    records, boundaries = native_sample()
    summary = aggregate_rates(
        records, "area", "loss", ["model_a", "model_b"], "exposure"
    )
    return prepare_geography(summary, boundaries, "area")


def map_axes(figure):
    return [axis for axis in figure.axes if axis.get_title()]


@pytest.mark.parametrize("ncols,facecolor", [(1, "white"), (2, "#2b303b")])
def test_static_comparison_scales_layers_opacity_contrast_and_native_export(
    tmp_path, ncols, facecolor
):
    mapped = sample()
    before = mapped.copy(deep=True)
    style = dict(matplotlib.rcParams)
    metrics = ["loss", "model_a", "model_b", "model_a_difference", "model_a_ratio"]
    figure = plot_geography(
        mapped,
        metrics,
        geoid="area",
        include_support=True,
        options=PlotOptions(ncols=ncols, facecolor=facecolor, alpha=0.4),
    )
    axes = map_axes(figure)
    assert [axis.get_title() for axis in axes] == metrics + [
        "support_count",
        "total_weight",
    ]
    assert len(figure.axes) == 14  # seven maps and seven colorbars, no unused grid axes
    assert [
        (axis.collections[0].norm.vmin, axis.collections[0].norm.vmax) for axis in axes
    ] == [(0, 11)] * 3 + [(-1, 1), (0.8, 10 / 11), (1, 2), (1, 4)]
    np.testing.assert_allclose(axes[0].collections[0].get_array(), [10, 4, 0])
    np.testing.assert_allclose(axes[1].collections[0].get_array(), [11, 5, 0])
    for axis in axes:
        assert (
            len(axis.collections) == 2
        )  # finite and missing polygons, drawn once each
        assert sum(len(layer.get_paths()) for layer in axis.collections) == 4
        assert all(layer.get_alpha() == 0.4 for layer in axis.collections)
        assert axis.collections[1].get_hatch() == "///"
        assert axis.title.get_color() == ("black" if facecolor == "white" else "white")
        assert axis.get_legend().get_texts()[0].get_text() == "Missing / undefined"
    assert not np.array_equal(
        axes[0].collections[0].get_facecolor()[-1],
        axes[0].collections[1].get_facecolor()[0],
    )
    figure.savefig(tmp_path / "comparison.png")
    assert (tmp_path / "comparison.png").stat().st_size > 1000
    assert dict(matplotlib.rcParams) == style
    assert_geodataframe_equal(mapped, before)
    assert mapped.attrs == before.attrs
    plt.close(figure)


@pytest.mark.parametrize("autobin", [False, True])
def test_single_constant_and_entirely_missing_maps(autobin):
    import holoviews as hv
    from bokeh.models import ColorBar

    mapped = sample()
    for values in ([5, 5, 5, np.nan], [0, 0, 0, np.nan], [np.nan] * 4):
        constant = mapped.assign(loss=values)
        figure = plot_geography(
            constant,
            ["loss"],
            geoid="area",
            options=PlotOptions(autobin=autobin, n_bins=20),
        )
        axis = map_axes(figure)[0]
        assert sum(len(layer.get_paths()) for layer in axis.collections) == 4
        assert len(figure.axes) == (1 if np.isnan(values).all() else 2)
        assert axis.collections[-1].get_hatch() == "///"
        if not np.isnan(values).all():
            assert axis.collections[0].norm.vmin < axis.collections[0].norm.vmax
        plt.close(figure)
        layout = plot_geography(
            constant,
            ["loss"],
            geoid="area",
            options=PlotOptions(interactive=True, autobin=autobin, n_bins=20),
        )
        assert len(layout) == 1
        model = hv.render(layout, backend="bokeh")
        assert len(list(model.select({"type": ColorBar}))) == (
            0 if np.isnan(values).all() else 1
        )


@pytest.mark.parametrize("autobin", [False, True])
def test_interactive_matches_static_scales_classes_hover_and_export(tmp_path, autobin):
    import holoviews as hv
    from bokeh.models import ColorBar, GlyphRenderer, HoverTool
    from bokeh.plotting import figure as BokehFigure

    mapped = sample()
    before = mapped.copy(deep=True)
    metrics = [
        "loss",
        "model_a",
        "model_b",
        "model_a_difference",
        "model_b_difference",
        "model_a_ratio",
    ]
    options = PlotOptions(
        autobin=autobin, n_bins=20, ncols=1, alpha=0.6, facecolor="#2b303b"
    )
    figure = plot_geography(
        mapped, metrics, geoid="area", include_support=True, options=options
    )
    options.interactive = True
    layout = plot_geography(
        mapped, metrics, geoid="area", include_support=True, options=options
    )
    assert isinstance(layout, hv.Layout)
    assert layout._max_cols == 1
    assert len(layout) == 8
    theme = hv.renderer("bokeh").theme
    model = hv.render(layout, backend="bokeh")
    assert hv.renderer("bokeh").theme is theme
    native_figures = list(model.select({"type": BokehFigure}))
    axes = map_axes(figure)
    for panel, axis in zip(layout.values(), axes):
        color = panel.opts.get("style", backend="bokeh").kwargs["color"].dimension.name
        actual = panel.data[color].to_numpy(dtype=float)
        expected = axis.collections[0].get_array()
        np.testing.assert_allclose(actual[~np.isnan(actual)], expected)
        np.testing.assert_allclose(
            panel.data[axis.get_title()], mapped[axis.get_title()], equal_nan=True
        )
        native = next(
            item
            for item in native_figures
            if item.title.text.startswith(axis.get_title() + " (")
            or item.title.text == axis.get_title()
        )
        bars = list(native.select({"type": ColorBar}))
        assert len(bars) == 1
        mapper = bars[0].color_mapper
        assert (mapper.low, mapper.high) == (
            axis.collections[0].norm.vmin,
            axis.collections[0].norm.vmax,
        )
        assert mapper.nan_color == "#bdbdbd"
        assert native.title.text_color == "white"
        assert native.border_fill_color == "#2b303b"
        glyphs = list(native.select({"type": GlyphRenderer}))
        assert len(glyphs) == 1
        assert glyphs[0].glyph.fill_alpha == 0.6
        hover = list(native.select({"type": HoverTool}))[0]
        assert {label for label, _ in hover.tooltips} == {
            "area",
            axis.get_title(),
            "support_count",
            "total_weight",
        }
        assert all("@{" in field for _, field in hover.tooltips)
    if autobin:
        # Pooled Fisher-Jenks classes include a true zero, not the missing area.
        np.testing.assert_allclose(
            layout.values()[0].data["_color_class"], [5, 3, 0, np.nan]
        )
        assert (
            axes[0].collections[0].norm.boundaries.tolist()
            == axes[1].collections[0].norm.boundaries.tolist()
        )
        # Symmetric difference bins put zero in the neutral middle class.
        diff_axis = axes[3]
        zero_class = layout.values()[3].data.loc[2, "_color_class"]
        colors = diff_axis.collections[0].cmap.colors
        assert zero_class == len(colors) // 2
    hv.save(layout, tmp_path / "comparison.html", backend="bokeh")
    assert (tmp_path / "comparison.html").stat().st_size > 1000
    assert_geodataframe_equal(mapped, before)
    assert mapped.attrs == before.attrs
    plt.close(figure)


def test_collision_safe_support_and_hover_names():
    import holoviews as hv
    from bokeh.models import GlyphRenderer, HoverTool

    from examples.native_comparison import sample as native_sample

    records, boundaries = native_sample()
    records = records.rename(
        columns={
            "loss": "support_count",
            "model_a": "_color_class",
            "model_b": "rate with spaces",
        }
    )
    mapped = prepare_geography(
        aggregate_means(
            records,
            "area",
            ["support_count", "_color_class", "rate with spaces"],
            "exposure",
        ),
        boundaries,
        "area",
    )
    mapped = mapped.rename(columns={"area": "color"})
    mapped["rate_with_spaces"] = mapped["rate with spaces"] + 1
    options = PlotOptions(interactive=True, autobin=True)
    layout = plot_geography(
        mapped,
        ["_color_class", "rate with spaces", "rate_with_spaces"],
        geoid="color",
        include_support=True,
        options=options,
    )
    assert len(layout) == 5
    assert layout._max_cols == 2
    assert mapped.attrs["support"]["count"] == "support_count_"
    assert "_color_class_" in layout.values()[0].data
    np.testing.assert_allclose(
        layout.values()[0].data["_color_class"], mapped["_color_class"], equal_nan=True
    )
    model = hv.render(layout, backend="bokeh")
    for hover in model.select({"type": HoverTool}):
        renderer = hover.renderers[0]
        assert isinstance(renderer, GlyphRenderer)
        fields = renderer.data_source.data.keys()
        for _, field in hover.tooltips:
            assert field[2:-1] in fields


def test_static_import_and_plot_do_not_load_interactive_or_change_style():
    script = """
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
before = dict(matplotlib.rcParams)
from examples.native_comparison import sample
from geomapviz import aggregate_means, prepare_geography
from geomapviz.plot import PlotOptions, plot_geography
records, boundaries = sample()
mapped = prepare_geography(aggregate_means(records, "area", ["model_a"], "exposure"), boundaries, "area")
figure = plot_geography(mapped, ["model_a"], geoid="area")
assert dict(matplotlib.rcParams) == before
optional = [name for name in ("holoviews", "geoviews", "bokeh", "panel", "cartopy", "seaborn", "contextily") if name in sys.modules]
print(json.dumps(optional))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, text=True, capture_output=True
    )
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "missing_crs",
        "missing_metric",
        "duplicate_metric",
        "empty_metrics",
        "nonnumeric",
        "infinite",
        "missing_support",
        "id_metric",
    ],
)
def test_invalid_prepared_input_fails_explicitly(problem):
    mapped = sample()
    metrics, include_support = ["loss"], False
    if problem == "empty":
        mapped = mapped.iloc[:0]
    elif problem == "missing_crs":
        mapped = mapped.set_crs(None, allow_override=True)
    elif problem == "missing_metric":
        metrics = ["absent"]
    elif problem == "duplicate_metric":
        metrics = ["loss", "loss"]
    elif problem == "empty_metrics":
        metrics = []
    elif problem == "nonnumeric":
        mapped = mapped.assign(loss="text")
    elif problem == "infinite":
        mapped = mapped.assign(loss=[np.inf, 1, 0, np.nan])
    elif problem == "id_metric":
        metrics = ["area"]
    else:
        mapped.attrs = {}
        include_support = True
    with pytest.raises((ValueError, TypeError)):
        plot_geography(mapped, metrics, geoid="area", include_support=include_support)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ncols": 0},
        {"ncols": True},
        {"n_bins": 0},
        {"alpha": -1},
        {"alpha": np.nan},
        {"figsize": (0, 8)},
        {"interactive": 1},
    ],
)
def test_invalid_options_fail_before_rendering(kwargs):
    with pytest.raises((ValueError, TypeError)):
        PlotOptions(**kwargs)


def test_package_rates_match_the_independent_native_m1_baseline():
    from examples.native_comparison import native_summary, sample

    records, _ = sample()
    pd.testing.assert_frame_equal(
        aggregate_rates(records, "area", "loss", ["model_a", "model_b"], "exposure"),
        native_summary(records),
    )


def test_close_classification_edges_remain_distinguishable():
    mapped = sample().assign(loss=[1.0, 1.000000001, 1.000000002, np.nan])
    figure = plot_geography(
        mapped, ["loss"], geoid="area", options=PlotOptions(autobin=True)
    )
    labels = [tick.get_text() for tick in figure.axes[1].get_yticklabels()]
    assert "1 < x ≤ 1.000000001" in labels
    assert "1.000000001 < x ≤ 1.000000002" in labels
    plt.close(figure)
