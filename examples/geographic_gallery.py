"""Real boundaries, simulated signals and published CBS population counts.

Run every static example (data paths are relative to this script):
    uv run --no-project --python 3.12 --with 'geomapviz==2.0.1' \
        geographic_gallery.py --output output
Add --interactive with geomapviz[interactive] for standalone, offline HTML.
"""

import argparse
import gc
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geomapviz import (  # noqa: E402
    aggregate_means,
    aggregate_rates,
    assign_parent,
    prepare_geography,
)
from geomapviz.plot import PlotOptions, plot_geography  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
METRICS = ["simulated_signal", "simulated_local", "simulated_east", "simulated_north"]
MODELS = ["simulated_model_east", "simulated_model_north"]
MISSING_MUNICIPALITY = "11002"  # Antwerpen: intentionally no observations.
ZERO_MUNICIPALITY = "21004"  # Bruxelles / Brussel: genuine observed zero.
UNDEFINED_RATIO_MUNICIPALITY = "52011"  # Charleroi: model east predicts zero.


# --8<-- [start:load]
def load_boundaries(country):
    filename = {
        "belgium": "belgium_municipalities_2024.zip",
        "netherlands": "netherlands_postcode4_2024.zip",
    }[country]
    return gpd.read_file(DATA / filename)


# --8<-- [end:load]


# --8<-- [start:simulation]
def simulate(boundaries, geoid):
    """Multiple records per area; all values and weights here are simulated."""
    rng = np.random.default_rng(2024)
    centers = boundaries.geometry.centroid  # Coordinates are projected metres.
    east = (centers.x - centers.x.min()) / (centers.x.max() - centers.x.min())
    north = (centers.y - centers.y.min()) / (centers.y.max() - centers.y.min())
    spatial = 2 + 0.7 * np.sin(3 * np.pi * east) + 0.5 * np.cos(2 * np.pi * north)
    areas = pd.DataFrame(
        {geoid: boundaries[geoid], "east": east, "north": north, "spatial": spatial}
    )
    # Unequal numbers of records make averaging municipality means misleading.
    records = areas.loc[
        areas.index.repeat(rng.integers(5, 16, len(areas)))
    ].reset_index(drop=True)
    noise = rng.normal(0, 0.15, len(records))
    records["simulated_signal"] = records.spatial + noise
    records["simulated_local"] = records.spatial + rng.normal(0, 0.3, len(records))
    records["simulated_east"] = records.spatial + 0.8 * (records.east - 0.5)
    records["simulated_north"] = records.spatial + 0.8 * (records.north - 0.5)
    records["exposure"] = rng.uniform(0.2, 3, len(records))  # Strictly positive.
    # Poisson observations are counts; aggregate_rates sums these only once.
    records["simulated_observed"] = rng.poisson(records.spatial * records.exposure)
    records[MODELS[0]] = records.spatial * (0.7 + 0.6 * records.east)
    records[MODELS[1]] = records.spatial * (0.7 + 0.6 * records.north)
    return records


# --8<-- [end:simulation]


def save_summary(mapped, geoid, output, name):
    mapped.drop(columns=mapped.geometry.name).to_csv(
        output / f"{name}.csv", index=False
    )
    metadata = {"geoid": geoid, "crs": str(mapped.crs), **mapped.attrs}
    (output / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")


# --8<-- [start:export]
def geographic_ranges(plot, _element):
    from bokeh.models import DataRange1d

    # Bokeh enforces match_aspect only with native auto ranges, not fixed ranges.
    plot.state.x_range = DataRange1d()
    plot.state.y_range = DataRange1d()


def export(
    mapped,
    metrics,
    geoid,
    output,
    name,
    interactive,
    *,
    classified=False,
    limits=None,
):
    options = PlotOptions(autobin=classified, n_bins=5, ncols=2, figsize=(16, 11))
    if len(metrics) == 1:
        options.figsize = (10, 8)
    figure = plot_geography(mapped, metrics, geoid=geoid, options=options)
    if limits is not None:
        # Native Matplotlib normalization shares a scale across geographic levels.
        for axis in figure.axes:
            for collection in axis.collections:
                collection.set_clim(*limits)
    figure.savefig(output / f"{name}.png", dpi=120)
    plt.close(figure)
    del figure
    gc.collect()  # Release full-detail figure cycles before the next export.
    if interactive:
        import geoviews as gv
        import holoviews as hv

        options.interactive = True
        layout = plot_geography(mapped, metrics, geoid=geoid, options=options)
        if limits is not None:
            layout = layout.map(lambda panel: panel.opts(clim=limits), gv.Polygons)

        # plot_geography already projects to Web Mercator. Native HoloViews
        # polygons retain data/options and avoid repeating Cartopy projection.
        def native_polygon(panel):
            rendering = {
                **panel.opts.get("plot").kwargs,
                **panel.opts.get("style").kwargs,
            }
            rendering.pop("width")
            rendering.pop("height")
            rendering["hooks"] = [*rendering["hooks"], geographic_ranges]
            return (
                panel.clone(new_type=hv.Polygons)
                .opts.clear()
                .opts(**rendering, data_aspect=1, responsive="width", aspect=1)
            )

        layout = layout.map(native_polygon, gv.Polygons).opts(
            sizing_mode="stretch_width"
        )
        hv.save(layout, output / f"{name}.html", backend="bokeh", resources="inline")
        del layout
        gc.collect()


# --8<-- [end:export]


# --8<-- [start:belgium]
def belgium(output, interactive):
    boundaries = load_boundaries("belgium")
    records = simulate(boundaries, "mun_id")
    records.to_csv(output / "belgium_records.csv", index=False)
    equal = aggregate_means(records, "mun_id", METRICS)
    weighted = aggregate_means(records, "mun_id", METRICS, weight="exposure")
    mapped = prepare_geography(weighted, boundaries, "mun_id")
    save_summary(mapped, "mun_id", output, "belgium_weighted")
    equal.to_csv(output / "belgium_equal.csv", index=False)
    export(mapped, ["simulated_signal"], "mun_id", output, "belgium_mean", interactive)
    export(mapped, METRICS, "mun_id", output, "belgium_predictors", interactive)
    export(
        mapped,
        METRICS,
        "mun_id",
        output,
        "belgium_classified",
        interactive,
        classified=True,
    )


# --8<-- [end:belgium]


# --8<-- [start:aggregation]
def aggregation(output, interactive):
    boundaries = load_boundaries("belgium")
    records = simulate(boundaries, "mun_id")
    levels = []
    for geoid in ("mun_id", "arr_id", "reg_id"):
        # Administrative assignment and geometry dissolution are separate steps.
        original = (
            records
            if geoid == "mun_id"
            else assign_parent(records, boundaries, "mun_id", geoid)
        )
        geometry = (
            boundaries
            if geoid == "mun_id"
            else boundaries[[geoid, "geometry"]].dissolve(geoid).reset_index()
        )
        summary = aggregate_means(original, geoid, ["simulated_signal"], "exposure")
        mapped = prepare_geography(summary, geometry, geoid)
        save_summary(mapped, geoid, output, f"aggregation_{geoid}")
        levels.append((geoid, mapped))
    values = pd.concat([frame.simulated_signal for _, frame in levels])
    limits = (float(values.min()), float(values.max()))
    for geoid, mapped in levels:
        export(
            mapped,
            ["simulated_signal"],
            geoid,
            output,
            f"aggregation_{geoid}",
            interactive,
            limits=limits,
        )


# --8<-- [end:aggregation]


# --8<-- [start:rates]
def rate_records(boundaries):
    records = simulate(boundaries, "mun_id")
    records = records.loc[records.mun_id != MISSING_MUNICIPALITY].copy()
    records.loc[records.mun_id == ZERO_MUNICIPALITY, "simulated_observed"] = 0
    records.loc[records.mun_id == UNDEFINED_RATIO_MUNICIPALITY, MODELS[0]] = 0
    return records


def rates(output, interactive):
    boundaries = load_boundaries("belgium")
    records = rate_records(boundaries)
    records.to_csv(output / "rates_records.csv", index=False)
    summary = aggregate_rates(
        records, "mun_id", "simulated_observed", MODELS, "exposure"
    )
    mapped = prepare_geography(summary, boundaries, "mun_id")
    save_summary(mapped, "mun_id", output, "rates")
    export(
        mapped,
        ["simulated_observed"] + MODELS,
        "mun_id",
        output,
        "rates_shared",
        interactive,
    )
    for kind in ("difference", "ratio"):
        metrics = [summary.attrs["comparisons"][model][kind] for model in MODELS]
        export(mapped, metrics, "mun_id", output, f"rates_{kind}", interactive)
    export(
        mapped,
        list(summary.attrs["support"].values()),
        "mun_id",
        output,
        "rates_support",
        interactive,
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    maximum = float(summary[["simulated_observed"] + MODELS].max().max())
    for axis, model in zip(axes, MODELS):
        axis.scatter(summary[model], summary.simulated_observed, s=8, alpha=0.5)
        axis.plot([0, maximum], [0, maximum], color="black", linewidth=1)
        axis.set(
            xlabel="Simulated predicted rate",
            ylabel="Simulated observed rate",
            title=model,
            xlim=(0, maximum),
            ylim=(0, maximum),
            aspect="equal",
        )
    figure.savefig(output / "rates_scatter.png", dpi=120)
    plt.close(figure)


# --8<-- [end:rates]


# --8<-- [start:netherlands]
def netherlands(output, interactive):
    boundaries = load_boundaries("netherlands")
    records = simulate(boundaries, "postcode")
    records.to_csv(output / "netherlands_records.csv", index=False)
    summary = aggregate_means(records, "postcode", METRICS, "exposure")
    mapped = prepare_geography(summary, boundaries, "postcode")
    save_summary(mapped, "postcode", output, "netherlands")
    export(mapped, METRICS, "postcode", output, "netherlands_continuous", interactive)
    export(
        mapped,
        ["simulated_signal"],
        "postcode",
        output,
        "netherlands_classified",
        interactive,
        classified=True,
    )


# --8<-- [end:netherlands]


# --8<-- [start:demographics]
def population_cohort():
    published = pd.read_csv(
        DATA / "netherlands_population_2024.csv", dtype={"postcode": "string"}
    )
    counts = ["residents", "under15", "age65plus"]
    # CBS: -99997 = 0–4 / suppressed / absent; -99995 = not yet published.
    published[counts] = published[counts].mask(published[counts].lt(0))
    valid = published[counts].notna().all(axis=1) & published.residents.gt(0)
    cohort = published.loc[valid].copy()
    cohort["under15_percent"] = 100 * cohort.under15 / cohort.residents
    cohort["age65plus_percent"] = 100 * cohort.age65plus / cohort.residents
    return published, cohort


def demographics(output, interactive):
    boundaries = load_boundaries("netherlands")
    published, cohort = population_cohort()
    published.assign(in_valid_cohort=published.postcode.isin(cohort.postcode)).to_csv(
        output / "demographics_cohort.csv", index=False
    )
    metrics = ["under15_percent", "age65plus_percent"]
    summary = aggregate_means(cohort, "postcode", metrics, weight="residents")
    mapped = prepare_geography(summary, boundaries, "postcode")
    save_summary(mapped, "postcode", output, "demographics")
    export(mapped, metrics, "postcode", output, "demographics", interactive)


# --8<-- [end:demographics]


CASES = {
    "belgium": belgium,
    "aggregation": aggregation,
    "rates": rates,
    "netherlands": netherlands,
    "demographics": demographics,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument(
        "--interactive", action="store_true", help="also export standalone offline HTML"
    )
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in CASES if args.case == "all" else [args.case]:
        CASES[name](args.output, args.interactive)
        print(f"{name}: exported to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
