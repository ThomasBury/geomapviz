"""Check numerical caller changes; layout/scale repairs remain in M4."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import box  # noqa: E402

from geomapviz.aggregator import dissolve_and_aggregate  # noqa: E402
from geomapviz import aggregate_rates  # noqa: E402
from geomapviz.plot import (  # noqa: E402
    PlotOptions,
    spatial_average_plot,
    spatial_average_facetplot,
)


def test_existing_mean_renderers_consume_the_new_contract():
    records = pd.DataFrame(
        {
            "area": pd.Categorical(["001", "001", "002"]),
            "avg": [10.0, 20.0, 0.0],
            "count": [8.0, 12.0, 0.0],
            "weight": [1.0, 3.0, 1.0],
        }
    )
    boundaries = gpd.GeoDataFrame(
        {"area": ["001", "002"]},
        geometry=[box(4, 50, 4.1, 50.1), box(4.1, 50, 4.2, 50.1)],
        crs="EPSG:4326",
    )
    original = records.copy(deep=True)
    original_geometry = boundaries.copy(deep=True)
    prepared = dissolve_and_aggregate(
        records, "avg", ["count"], geoid="area", weight="weight", shp_file=boundaries
    )
    assert set(prepared["model"]) == {"avg", "count"}
    assert (
        prepared.loc[
            (prepared["area"] == "001") & (prepared["model"] == "avg"), "avg"
        ].item()
        == 17.5
    )
    assert not {"ci_low", "ci_up"} & set(prepared.columns)
    options = PlotOptions(
        records,
        "avg",
        other_cols_avg=["count"],
        weight="weight",
        geoid="area",
        shp_file=boundaries,
        normalize=False,
    )
    for figure in (spatial_average_plot(options), spatial_average_facetplot(options)):
        assert isinstance(figure, matplotlib.figure.Figure)
        assert any(axis.collections for axis in figure.axes)
        plt.close(figure)
    pd.testing.assert_frame_equal(records, original)
    pd.testing.assert_frame_equal(boundaries, original_geometry)


def test_package_rates_match_the_independent_native_m1_baseline():
    from examples.native_comparison import native_summary, sample

    records, _ = sample()
    pd.testing.assert_frame_equal(
        aggregate_rates(records, "area", "loss", ["model_a", "model_b"], "exposure"),
        native_summary(records),
    )
