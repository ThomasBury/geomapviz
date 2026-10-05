"""Synthetic checks for boundary joins, parent arithmetic and projections."""

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from geomapviz import aggregate_means, aggregate_rates, assign_parent, prepare_geography
from geomapviz.aggregator import dissolve_and_aggregate
from geomapviz.shapefiles import load_geometry


def sample():
    from examples.native_comparison import sample as native_sample

    records, boundaries = native_sample()
    boundaries["region"] = pd.Categorical(["north", "north", "south", "empty"])
    return records, boundaries


def test_coverage_crs_geometry_metadata_and_inputs_are_preserved():
    records, boundaries = sample()
    boundaries = boundaries.rename_geometry("shape")
    boundaries.index = [9, 3, 7, 5]
    original = boundaries.copy(deep=True)
    summary = aggregate_rates(records, "area", "loss", ["model_a"], "exposure")
    summary.index = [8, 2, 1]
    before = summary.copy(deep=True)
    mapped = prepare_geography(summary, boundaries, "area")
    assert isinstance(mapped, gpd.GeoDataFrame)
    assert mapped.crs == boundaries.crs
    assert mapped.geometry.name == "shape"
    assert mapped["area"].tolist() == ["001", "002", "003", "004"]
    np.testing.assert_allclose(mapped["loss"], [10, 4, 0, np.nan])
    assert mapped.attrs["coverage"] == {
        "matched": ["001", "002", "003"],
        "unmatched": ["004"],
    }
    for key in summary.attrs:
        assert mapped.attrs[key] == summary.attrs[key]
    assert mapped.loc[3, summary.columns.drop("area")].isna().all()
    assert mapped.geometry.to_list() == boundaries.geometry.to_list()
    assert_geodataframe_equal(boundaries, original)
    pd.testing.assert_frame_equal(summary, before)


@pytest.mark.parametrize(
    "ids", [[1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], pd.Categorical(["1", "2", "3", "4"])]
)
def test_numeric_and_categorical_ids_match_without_guessing_leading_zeros(ids):
    _, boundaries = sample()
    boundaries["area"] = ids
    summary = pd.DataFrame(
        {"area": pd.Categorical(["1", "2", "3"]), "value": [0, 2, 3]}
    )
    mapped = prepare_geography(summary, boundaries, "area")
    assert mapped["area"].tolist() == ["1", "2", "3", "4"]
    assert mapped.loc[0, "value"] == 0
    summary["area"] = [1.0, 2.0, 3.0]
    assert prepare_geography(summary, boundaries, "area").loc[0, "value"] == 0
    summary["area"] = ["001", "2", "3"]
    with pytest.raises(ValueError, match="001"):
        prepare_geography(summary, boundaries, "area")


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "normalized_duplicate",
        "missing_id",
        "missing_column",
        "missing_crs",
        "missing_geometry",
        "empty_geometry",
        "invalid_geometry",
        "empty",
        "duplicate_column",
        "no_active_geometry",
    ],
)
def test_bad_boundaries_fail_before_joining(problem):
    records, boundaries = sample()
    summary = aggregate_means(records, "area", ["model_a"], "exposure")
    if problem == "duplicate":
        boundaries.loc[1, "area"] = "001"
    elif problem == "normalized_duplicate":
        boundaries["area"] = [1, "1", 3, 4]
    elif problem == "missing_id":
        boundaries.loc[1, "area"] = None
    elif problem == "missing_column":
        boundaries = boundaries.drop(columns="area")
    elif problem == "missing_crs":
        boundaries = boundaries.set_crs(None, allow_override=True)
    elif problem == "missing_geometry":
        boundaries.loc[1, "geometry"] = None
    elif problem == "empty_geometry":
        boundaries.loc[1, "geometry"] = Polygon()
    elif problem == "invalid_geometry":
        boundaries.loc[1, "geometry"] = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    elif problem == "empty":
        boundaries = boundaries.iloc[:0]
    elif problem == "duplicate_column":
        boundaries = pd.concat([boundaries, boundaries[["area"]]], axis=1)
    else:
        boundaries = gpd.GeoDataFrame({"area": ["001"], "shape": [box(0, 0, 1, 1)]})
    with pytest.raises(ValueError):
        prepare_geography(summary, boundaries, "area")


def test_summary_validation_unknown_ids_and_collisions():
    records, boundaries = sample()
    summary = aggregate_means(records, "area", ["model_a"], "exposure")
    with pytest.raises(ValueError, match="Duplicate summary.*001"):
        prepare_geography(pd.concat([summary, summary.iloc[:1]]), boundaries, "area")
    unknown = summary.copy()
    unknown["area"] = ["unknown", "002", "003"]
    with pytest.raises(ValueError, match="unknown"):
        prepare_geography(unknown, boundaries, "area")
    with pytest.raises(ValueError, match="missing IDs"):
        prepare_geography(summary.assign(area=[None, "002", "003"]), boundaries, "area")
    with pytest.raises(ValueError, match="invalid ID"):
        prepare_geography(summary.assign(area=[np.inf, 2, 3]), boundaries, "area")
    with pytest.raises(ValueError, match="overlap.*model_a"):
        prepare_geography(summary, boundaries.assign(model_a=999), "area")
    with pytest.raises(ValueError, match="overlap.*geometry"):
        prepare_geography(summary.assign(geometry=42), boundaries, "area")
    with pytest.raises(ValueError, match="at least one"):
        prepare_geography(summary.iloc[:0], boundaries, "area")
    with pytest.raises(TypeError, match="GeoDataFrame"):
        prepare_geography(summary, pd.DataFrame(boundaries), "area")


@pytest.mark.parametrize("observed_kind", ["total", "rate"])
def test_parent_means_and_rates_reaggregate_records_not_area_averages(observed_kind):
    records, boundaries = sample()
    if observed_kind == "rate":
        records["loss"] = [10, 10, 4, 999, 0]
    records.index = [6, 6, 9, 2, 1]
    before, original_boundaries = records.copy(deep=True), boundaries.copy(deep=True)
    assigned = assign_parent(records, boundaries, "area", "region")
    assert assigned["region"].tolist() == ["north", "north", "north", "north", "south"]
    pd.testing.assert_frame_equal(assigned.drop(columns="region"), records)
    expected = records.assign(region=["north", "north", "north", "north", "south"])
    rates = aggregate_rates(
        assigned,
        "region",
        "loss",
        ["model_a", "model_b"],
        "exposure",
        observed_kind=observed_kind,
    )
    pd.testing.assert_frame_equal(
        rates,
        aggregate_rates(
            expected,
            "region",
            "loss",
            ["model_a", "model_b"],
            "exposure",
            observed_kind=observed_kind,
        ),
        check_dtype=False,
    )
    north = rates.set_index("region").loc["north"]
    assert north["loss"] == 8  # (40 + 8) / (4 + 2), not (10 + 4) / 2
    assert north["model_a"] == 9  # (44 + 10) / 6
    assert north["loss_total"] == 48
    assert north["model_a_total"] == 54
    assert north["model_a_difference"] == -1
    assert north["model_a_ratio"] == pytest.approx(8 / 9)
    assert north["support_count"] == 3
    assert north["total_weight"] == 6
    means = aggregate_means(assigned, "region", ["model_a"], "exposure")
    assert means.set_index("region").loc["north", "model_a"] == 9
    parents = (
        boundaries[["region", "geometry"]]
        .dissolve(by="region", observed=True)
        .reset_index()
    )
    mapped = prepare_geography(rates, parents, "region")
    assert mapped.attrs["coverage"] == {
        "matched": ["north", "south"],
        "unmatched": ["empty"],
    }
    assert mapped.loc[mapped["region"] == "empty", "loss"].isna().all()
    assert (
        mapped.loc[mapped["region"] == "north", "geometry"]
        .item()
        .equals(boundaries.geometry.iloc[:2].union_all())
    )
    pd.testing.assert_frame_equal(records, before)
    assert_geodataframe_equal(boundaries, original_boundaries)


def test_parent_mapping_rejects_ambiguity_missing_parents_and_conflicting_labels():
    records, boundaries = sample()
    with pytest.raises(ValueError, match="Duplicate boundary"):
        assign_parent(
            records,
            pd.concat([boundaries, boundaries.iloc[:1].assign(region="other")]),
            "area",
            "region",
        )
    with pytest.raises(ValueError, match="region.*missing"):
        assign_parent(
            records,
            boundaries.assign(region=["north", "north", "south", None]),
            "area",
            "region",
        )
    with pytest.raises(ValueError, match="missing ID.*absent"):
        assign_parent(records, boundaries, "area", "absent")
    with pytest.raises(ValueError, match="disagree.*001"):
        assign_parent(records.assign(region="wrong"), boundaries, "area", "region")
    with pytest.raises(ValueError, match="unknown"):
        assign_parent(records.assign(area="unknown"), boundaries, "area", "region")
    with pytest.raises(ValueError, match="distinct"):
        assign_parent(records, boundaries, "area", "area")
    assigned = assign_parent(records, boundaries, "area", "region")
    pd.testing.assert_frame_equal(
        assign_parent(assigned, boundaries, "area", "region"), assigned
    )


def test_legacy_adapter_keeps_missing_areas_for_each_metric_and_uses_parent_mapping():
    records, boundaries = sample()
    for parent in (None, "region"):
        prepared = dissolve_and_aggregate(
            records,
            "loss",
            ["model_a"],
            dissolve_on=parent,
            geoid="area",
            weight="exposure",
            shp_file=boundaries,
        )
        assert prepared.crs == boundaries.crs
        assert set(prepared["model"]) == {"loss", "model_a"}
        geoid, unmatched = ("region", "empty") if parent else ("area", "004")
        missing = prepared.loc[prepared[geoid] == unmatched]
        assert len(missing) == 2
        assert missing[["avg", "count", "weight"]].isna().all().all()
        assert prepared.attrs["coverage"]["unmatched"] == [unmatched]
        if parent:
            assert (
                prepared.loc[
                    (prepared[geoid] == "north") & (prepared["model"] == "model_a"),
                    "avg",
                ].item()
                == 9
            )
    with pytest.raises(ValueError, match="unknown"):
        dissolve_and_aggregate(
            records.assign(area="unknown"), "loss", geoid="area", shp_file=boundaries
        )


def test_file_loader_preserves_crs_and_interactive_renderer_transforms_coordinates(
    tmp_path,
):
    from geomapviz.plot import get_facet, get_interactive_plot_options, get_tiles

    _, boundaries = sample()
    filename = tmp_path / "boundaries.geojson"
    boundaries.to_file(filename, driver="GeoJSON")
    loaded = load_geometry(filename, "area")
    assert loaded.crs == boundaries.crs
    assert loaded["area"].tolist() == boundaries["area"].tolist()
    assert loaded.geometry.to_list() == boundaries.geometry.to_list()
    original = boundaries.copy(deep=True)
    options = get_interactive_plot_options("value", "viridis", 0.6)
    for source in (boundaries, boundaries.to_crs(epsg=31370)):
        mapped = prepare_geography(
            pd.DataFrame({"area": ["001"], "avg": [10.0]}), source, "area"
        )
        facet = get_facet(mapped, get_tiles(None), 0, 10, options)
        polygons = facet.values()[-1]
        expected = boundaries.to_crs(epsg=3857)
        np.testing.assert_allclose(
            polygons.data.total_bounds, expected.total_bounds, atol=0.1
        )
        assert polygons.data.crs.to_epsg() == 3857
        round_trip = polygons.data.to_crs(boundaries.crs)
        np.testing.assert_allclose(
            round_trip.total_bounds, boundaries.total_bounds, atol=1e-6
        )
        assert round_trip.geometry.geom_equals_exact(
            boundaries.geometry, tolerance=1e-6
        ).all()
    assert_geodataframe_equal(boundaries, original)
