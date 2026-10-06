"""M1 native baseline. Run with .venv/bin/python examples/native_comparison.py."""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import box

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sample():
    """Synthetic policy records; no claim of real-world user demand."""
    records = pd.DataFrame(
        {
            "area": pd.Categorical(["001", "001", "002", "002", "003"]),
            "loss": [10.0, 30.0, 8.0, 999.0, 0.0],
            "exposure": [1.0, 3.0, 2.0, 0.0, 1.0],
            "model_a": [8.0, 12.0, 5.0, 999.0, 0.0],
            "model_b": [10.0, 10.0, 3.0, 999.0, 1.0],
        }
    )
    boundaries = gpd.GeoDataFrame(
        {"area": ["001", "002", "003", "004"]},
        geometry=[box(4 + i * 0.1, 50, 4.08 + i * 0.1, 50.08) for i in range(4)],
        crs="EPSG:4326",
    )
    return records, boundaries


def native_summary(records):
    """Pandas implementation of the retained total/exposure task."""
    # ponytail: fixed example columns; use package validation for arbitrary inputs.
    metrics = ["loss", "model_a", "model_b"]
    if records.empty or records["area"].isna().any():
        raise ValueError("area must be present on a non-empty cohort")
    if not np.isfinite(records[metrics + ["exposure"]].to_numpy()).all():
        raise ValueError("Select a common finite cohort explicitly")
    if (records["exposure"] < 0).any():
        raise ValueError("exposure must be non-negative")
    positive = records.loc[records["exposure"] > 0].copy()
    unsupported = set(records["area"]) - set(positive["area"])
    if unsupported:
        raise ValueError(f"No positive exposure: {sorted(unsupported)}")
    for model in metrics[1:]:
        positive[model] *= positive["exposure"]
    grouped = positive.groupby("area", observed=True, sort=False)
    totals = grouped[metrics + ["exposure"]].sum()
    rates = totals[metrics].div(totals["exposure"], axis=0)
    rates["support_count"] = grouped.size()
    rates["total_weight"] = totals["exposure"]
    for metric in metrics:
        rates[f"{metric}_total"] = totals[metric]
    for model in metrics[1:]:
        rates[f"{model}_difference"] = rates["loss"] - rates[model]
        rates[f"{model}_ratio"] = totals["loss"] / totals[model].replace(0, np.nan)
    return rates.reset_index()


def prepare_map(summary, boundaries):
    if boundaries.crs is None:
        raise ValueError("Boundary CRS is required")
    if boundaries["area"].isna().any() or boundaries["area"].duplicated().any():
        raise ValueError("One non-missing boundary identifier per area is required")
    geometry = boundaries.geometry
    if geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all():
        raise ValueError("Non-empty valid geometries are required")
    unknown = set(summary["area"]) - set(boundaries["area"])
    if unknown:
        raise ValueError(f"Unknown observation IDs: {sorted(unknown)}")
    mapped = boundaries.merge(summary, on="area", how="left", validate="one_to_one")
    coverage = {
        "matched": mapped.loc[mapped["support_count"].notna(), "area"].tolist(),
        "unmatched": mapped.loc[mapped["support_count"].isna(), "area"].tolist(),
    }
    return mapped, coverage


def render(mapped, output):
    output.mkdir(parents=True, exist_ok=True)
    metrics = ["loss", "model_a", "model_b"]
    finite = mapped[metrics].to_numpy()
    limits = (np.nanmin(finite), np.nanmax(finite))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for metric, axis in zip(metrics, axes):
        mapped.plot(
            column=metric,
            ax=axis,
            vmin=limits[0],
            vmax=limits[1],
            legend=True,
            missing_kwds={"color": "lightgrey", "hatch": "///"},
        )
        axis.set_title(metric)
        axis.set_axis_off()
        assert (axis.collections[0].norm.vmin, axis.collections[0].norm.vmax) == limits
    figure.savefig(output / "native-static.png")
    plt.close(figure)

    import holoviews as hv
    import hvplot.pandas  # noqa: F401 - registers the GeoDataFrame accessor
    import cartopy.crs as ccrs

    panels = [
        mapped.hvplot.polygons(
            geo=True,
            crs=ccrs.PlateCarree(),
            c=metric,
            clim=limits,
            cmap="viridis",
            hover_cols=["area", "support_count", "total_weight"],
            title=metric,
            colorbar=True,
            width=350,
            height=300,
        )
        for metric in metrics
    ]
    hv.save(hv.Layout(panels).cols(3), output / "native-interactive.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/geomapviz-m1"))
    arguments = parser.parse_args()
    records, boundaries = sample()
    before = records.copy(deep=True)
    summary = native_summary(records)
    pd.testing.assert_frame_equal(records, before)
    np.testing.assert_allclose(summary["loss"], [10, 4, 0])
    np.testing.assert_allclose(summary["model_a"], [11, 5, 0])
    np.testing.assert_allclose(summary["model_a_difference"], [-1, -1, 0])
    np.testing.assert_allclose(summary["model_a_ratio"], [10 / 11, 0.8, np.nan])
    assert summary["support_count"].tolist() == [2, 1, 1]
    mapped, coverage = prepare_map(summary, boundaries)
    assert coverage == {"matched": ["001", "002", "003"], "unmatched": ["004"]}
    assert mapped.loc[mapped["area"] == "004", "loss"].isna().all()
    render(mapped, arguments.output)
    print(summary.to_string(index=False))
    print(coverage)
    print(f"Exported static PNG and interactive HTML to {arguments.output}")


if __name__ == "__main__":
    main()
