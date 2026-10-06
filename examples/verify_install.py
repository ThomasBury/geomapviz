"""Check a built installation from a copied examples directory outside the checkout.

Run with the installed environment's Python, optionally adding --interactive.
"""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import pandas as pd

import geomapviz
from geomapviz import aggregate_means, aggregate_rates, assign_parent, prepare_geography


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert Path(geomapviz.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert geomapviz.__version__ == metadata.version("geomapviz")
    frame = pd.DataFrame({"area": ["001", "001"], "value": [10, 20], "weight": [1, 3]})
    assert aggregate_means(frame, "area", ["value"], "weight").loc[0, "value"] == 17.5
    optional = {"holoviews", "geoviews", "cartopy", "bokeh", "panel"}
    assert not (optional | {"geopandas", "matplotlib"}).intersection(sys.modules)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    style = dict(matplotlib.rcParams)
    from native_comparison import native_summary, sample
    from geomapviz.plot import PlotOptions, plot_geography

    assert dict(matplotlib.rcParams) == style
    records, boundaries = sample()
    before = records.copy(deep=True)
    summary = aggregate_rates(
        records, "area", "loss", ["model_a", "model_b"], "exposure"
    )
    pd.testing.assert_frame_equal(summary, native_summary(records))
    mapped = prepare_geography(summary, boundaries, "area")
    assert mapped.attrs["coverage"] == {
        "matched": ["001", "002", "003"],
        "unmatched": ["004"],
    }
    assert pd.isna(mapped.loc[3, "loss"])
    parents = boundaries.assign(region=["north", "north", "south", "empty"])
    parent_records = assign_parent(records, parents, "area", "region")
    parent_summary = aggregate_rates(
        parent_records, "region", "loss", ["model_a", "model_b"], "exposure"
    )
    north = parent_summary.set_index("region").loc["north"]
    np.testing.assert_allclose(
        north[["loss_total", "total_weight", "loss", "model_a", "model_a_ratio"]],
        [48, 6, 8, 9, 8 / 9],
    )
    pd.testing.assert_frame_equal(records, before)
    metrics = ["loss", "model_a", "model_b", "model_a_difference", "model_a_ratio"]
    for autobin in (False, True):
        name = "classified" if autobin else "continuous"
        figure = plot_geography(
            mapped,
            metrics,
            geoid="area",
            include_support=True,
            options=PlotOptions(autobin=autobin, ncols=3, alpha=0.8),
        )
        path = args.output / f"{name}.png"
        figure.savefig(path)
        plt.close(figure)
        assert path.stat().st_size > 1000
    assert dict(matplotlib.rcParams) == style
    assert not optional.intersection(sys.modules)
    if args.interactive:
        import holoviews as hv
        from bokeh.themes import Theme

        renderer = hv.renderer("bokeh")
        theme = Theme(json={"attrs": {"Figure": {"background_fill_color": "#eeeeee"}}})
        renderer.theme = theme
        for autobin in (False, True):
            name = "classified" if autobin else "continuous"
            layout = plot_geography(
                mapped,
                metrics,
                geoid="area",
                include_support=True,
                options=PlotOptions(interactive=True, autobin=autobin, ncols=1),
            )
            assert renderer.theme is theme
            path = args.output / f"{name}.html"
            hv.save(layout, path, backend="bokeh", resources="inline")
            assert path.stat().st_size > 1000
        assert renderer.theme is theme
        assert dict(matplotlib.rcParams) == style
    else:
        installed = {dist.metadata["Name"].lower() for dist in metadata.distributions()}
        assert not optional.intersection(installed)
        try:
            plot_geography(
                mapped, ["loss"], geoid="area", options=PlotOptions(interactive=True)
            )
        except ImportError as error:
            assert "geomapviz[interactive]" in str(error)
        else:
            raise AssertionError("Interaction must require the extra")
    versions = {
        name: metadata.version(name)
        for name in (
            "geomapviz",
            "numpy",
            "pandas",
            "geopandas",
            "matplotlib",
            "mapclassify",
        )
    }
    if args.interactive:
        versions.update({name: metadata.version(name) for name in sorted(optional)})
    print(
        json.dumps(
            {
                "python": sys.version,
                "versions": versions,
                "package": geomapviz.__file__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
