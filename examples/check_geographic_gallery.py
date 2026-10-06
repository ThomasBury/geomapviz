"""Independent arithmetic and snapshot checks; no downloads or optional stack.

Run with the same base installation used for geographic_gallery.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import geographic_gallery as gallery
from geomapviz import aggregate_means, aggregate_rates, assign_parent, prepare_geography


def check_means(records, geoid):
    results = {}
    for weight in (None, "exposure"):
        summary = aggregate_means(records, geoid, gallery.METRICS, weight).set_index(
            geoid
        )
        for key, group in records.groupby(geoid):
            weights = np.ones(len(group)) if weight is None else group[weight]
            for metric in gallery.METRICS:
                expected = np.average(group[metric], weights=weights)
                np.testing.assert_allclose(summary.loc[key, metric], expected)
            assert summary.loc[key, "support_count"] == len(group)
            np.testing.assert_allclose(summary.loc[key, "total_weight"], sum(weights))
        results[weight] = summary
    return results


def check_rates(records, geoid):
    summary = aggregate_rates(
        records, geoid, "simulated_observed", gallery.MODELS, "exposure"
    ).set_index(geoid)
    for key, group in records.groupby(geoid):
        exposure = group.exposure.sum()
        observed = group.simulated_observed.sum()
        np.testing.assert_allclose(
            summary.loc[key, "simulated_observed"], observed / exposure
        )
        np.testing.assert_allclose(
            summary.loc[key, "simulated_observed_total"], observed
        )
        for model in gallery.MODELS:
            expected = np.dot(group[model], group.exposure)
            np.testing.assert_allclose(summary.loc[key, model], expected / exposure)
            np.testing.assert_allclose(summary.loc[key, f"{model}_total"], expected)
            np.testing.assert_allclose(
                summary.loc[key, f"{model}_difference"],
                (observed - expected) / exposure,
            )
            ratio = summary.loc[key, f"{model}_ratio"]
            if expected == 0:
                assert pd.isna(ratio)
            else:
                np.testing.assert_allclose(ratio, observed / expected)
        assert summary.loc[key, "support_count"] == len(group)
        np.testing.assert_allclose(summary.loc[key, "total_weight"], exposure)
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exports",
        type=Path,
        nargs="*",
        default=[],
        help="also verify runner CSVs in these output directories",
    )
    args = parser.parse_args()

    def check_export(name, expected, geoid, columns=None):
        columns = list(expected.columns) if columns is None else columns
        for folder in args.exports:
            actual = pd.read_csv(
                folder / f"{name}.csv", dtype={geoid: "string"}
            ).set_index(geoid)
            assert actual.index.is_unique and set(actual.index) == set(expected.index)
            np.testing.assert_allclose(
                actual.loc[expected.index, columns].to_numpy(dtype=float),
                expected[columns].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )

    manifest = json.loads((gallery.DATA / "manifest.json").read_text())
    for name, checksum in manifest["outputs"].items():
        assert (
            hashlib.sha256((gallery.DATA / name).read_bytes()).hexdigest() == checksum
        )
    for country, geoid, crs, filename, stem in (
        ("belgium", "mun_id", 31370, "belgium_municipalities_2024.zip", "belgium"),
        (
            "netherlands",
            "postcode",
            28992,
            "netherlands_postcode4_2024.zip",
            "postcode4",
        ),
    ):
        with ZipFile(gallery.DATA / filename) as archive:
            assert set(archive.namelist()) == {
                stem + suffix for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg")
            }
            assert archive.read(stem + ".cpg").decode().upper() == "UTF-8"
        boundaries = gallery.load_boundaries(country)
        assert boundaries.crs.to_epsg() == crs and boundaries.crs.is_projected
        assert boundaries[geoid].is_unique and boundaries[geoid].notna().all()
        assert boundaries[geoid].map(lambda value: isinstance(value, str)).all()
        assert boundaries.is_valid.all() and not boundaries.is_empty.any()
        assert boundaries.geom_type.isin(["Polygon", "MultiPolygon"]).all()
        assert (
            len(boundaries)
            == manifest[country][
                "municipalities" if country == "belgium" else "postcodes"
            ]
        )
        records = gallery.simulate(boundaries, geoid)
        assert records.exposure.gt(0).all()
        pd.testing.assert_frame_equal(records, gallery.simulate(boundaries, geoid))
        means = check_means(records, geoid)
        check_export(
            "belgium_weighted" if country == "belgium" else "netherlands",
            means["exposure"],
            geoid,
        )
        summary = aggregate_means(records, geoid, gallery.METRICS, "exposure")
        mapped = prepare_geography(summary, boundaries, geoid)
        assert mapped.attrs["coverage"]["unmatched"] == []
        if country == "belgium":
            check_export("belgium_equal", means[None], geoid)
            parent_columns = ["simulated_signal", "support_count", "total_weight"]
            check_export("aggregation_mun_id", means["exposure"], geoid, parent_columns)
            rates = gallery.rate_records(boundaries)
            for parent, count in (("arr_id", 43), ("reg_id", 3)):
                assert (
                    boundaries[parent].notna().all()
                    and boundaries[parent].nunique() == count
                )
                # Check original-record parent assignments against the explicit source table.
                original = assign_parent(records, boundaries, "mun_id", parent)
                lookup = boundaries.set_index("mun_id")[parent]
                assert original[parent].eq(original.mun_id.map(lookup)).all()
                parent_means = check_means(original, parent)
                check_export(
                    f"aggregation_{parent}",
                    parent_means["exposure"],
                    parent,
                    parent_columns,
                )
                parent_rates = assign_parent(rates, boundaries, "mun_id", parent)
                parent_summary = check_rates(parent_rates, parent)
                dissolved = (
                    boundaries[[parent, "geometry"]].dissolve(parent).reset_index()
                )
                assert dissolved.crs == boundaries.crs and dissolved.is_valid.all()
                assert abs(dissolved.area.sum() - boundaries.area.sum()) < 1
                assert (
                    prepare_geography(parent_summary, dissolved, parent).attrs[
                        "coverage"
                    ]["unmatched"]
                    == []
                )
            summary = check_rates(rates, "mun_id")
            mapped = prepare_geography(summary, boundaries, "mun_id").set_index(
                "mun_id"
            )
            check_export("rates", mapped[list(summary.columns[1:])], geoid)
            assert mapped.attrs["coverage"]["unmatched"] == [
                gallery.MISSING_MUNICIPALITY
            ]
            assert pd.isna(
                mapped.loc[gallery.MISSING_MUNICIPALITY, "simulated_observed"]
            )
            assert mapped.loc[gallery.ZERO_MUNICIPALITY, "simulated_observed"] == 0
            assert mapped.loc[gallery.ZERO_MUNICIPALITY, "total_weight"] > 0
            assert pd.isna(
                mapped.loc[
                    gallery.UNDEFINED_RATIO_MUNICIPALITY, gallery.MODELS[0] + "_ratio"
                ]
            )
            assert (
                mapped.loc[gallery.UNDEFINED_RATIO_MUNICIPALITY, gallery.MODELS[0]] == 0
            )
        else:
            published, cohort = gallery.population_cohort()
            assert published.postcode.is_unique and set(published.postcode) == set(
                boundaries.postcode
            )
            assert published.year.eq(2024).all()
            assert len(cohort) == manifest[country]["valid_population_cohort"]
            assert len(published) - len(cohort) == 103
            assert cohort[["under15_percent", "age65plus_percent"]].ge(0).all().all()
            assert cohort[["under15_percent", "age65plus_percent"]].le(100).all().all()
            np.testing.assert_allclose(
                cohort.under15_percent, 100 * cohort.under15 / cohort.residents
            )
            np.testing.assert_allclose(
                cohort.age65plus_percent, 100 * cohort.age65plus / cohort.residents
            )
            summary = aggregate_means(
                cohort, geoid, ["under15_percent", "age65plus_percent"], "residents"
            )
            mapped = prepare_geography(summary, boundaries, geoid)
            check_export(
                "demographics",
                mapped.set_index(geoid)[list(summary.columns[1:])],
                geoid,
            )
            assert len(mapped.attrs["coverage"]["unmatched"]) == 103
            assert mapped.under15_percent.isna().sum() == 103
        print(
            f"{country}: snapshot, coverage and independent arithmetic pass", flush=True
        )


if __name__ == "__main__":
    main()
