import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from geomapviz import aggregate_means, aggregate_rates


def test_weighted_and_unweighted_means_preserve_cohort_and_names():
    frame = pd.DataFrame(
        {
            "area": pd.Categorical(
                ["001", "001", "002", "002"], categories=["002", "001", "unused"]
            ),
            "count": [10, 20, 0, 1000],
            "weight": [1, 3, 2, 0],
            "support_count": [2, 4, 6, 8],
            "total_weight": [5, 9, 1, 1000],
        },
        index=[7, 7, 2, 1],
    )
    original = frame.copy(deep=True)
    metrics = ["count", "weight", "support_count", "total_weight"]
    result = aggregate_means(frame, "area", metrics, "weight")
    assert isinstance(result, pd.DataFrame)
    assert result["area"].tolist() == ["001", "002"]
    assert isinstance(result["area"].dtype, pd.CategoricalDtype)
    np.testing.assert_allclose(result["count"], [17.5, 0])
    np.testing.assert_allclose(result["weight"], [2.5, 2])
    np.testing.assert_allclose(result["support_count"], [3.5, 6])
    np.testing.assert_allclose(result["total_weight"], [8, 1])
    assert result.attrs["support"] == {
        "count": "support_count_",
        "weight": "total_weight_",
    }
    assert result["support_count_"].tolist() == [2, 1]
    assert result["total_weight_"].tolist() == [4, 2]
    equal = aggregate_means(frame, "area", ["count"])
    np.testing.assert_allclose(equal["count"], [15, 500])
    assert equal["support_count"].tolist() == [2, 2]
    assert equal["total_weight"].tolist() == [2, 2]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("column", ["observed", "prediction", "exposure"])
@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf])
def test_invalid_values_are_rejected_on_the_common_cohort(column, invalid):
    frame = pd.DataFrame(
        {
            "id": ["001", "001"],
            "observed": [10.0, 20.0],
            "prediction": [8.0, 12.0],
            "exposure": [1.0, 0.0],
        }
    )
    frame.loc[1, column] = invalid
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match=column):
        aggregate_means(frame, "id", ["observed", "prediction"], "exposure")
    with pytest.raises(ValueError, match=column):
        aggregate_rates(frame, "id", "observed", ["prediction"], "exposure")
    pd.testing.assert_frame_equal(frame, original)


def test_negative_weights_and_unsupported_areas_fail():
    frame = pd.DataFrame(
        {"id": ["001", "002"], "value": [10.0, 20.0], "w": [1.0, -1.0]}
    )
    with pytest.raises(ValueError, match="w.*negative"):
        aggregate_means(frame, "id", ["value"], "w")
    frame["w"] = [1.0, 0.0]
    with pytest.raises(ValueError, match="002"):
        aggregate_means(frame, "id", ["value"], "w")
    frame["p"] = [8.0, 12.0]
    with pytest.raises(ValueError, match="002"):
        aggregate_rates(frame, "id", "value", ["p"], "w")


def test_rates_match_hand_values_for_observed_totals_and_existing_rates():
    frame = pd.DataFrame(
        {
            "area": pd.Categorical(["001", "001", "002", "002", "003"]),
            "loss": [10.0, 30.0, 8.0, 999.0, 0.0],
            "exposure": [1.0, 3.0, 2.0, 0.0, 1.0],
            "model_a": [8.0, 12.0, 5.0, 999.0, 0.0],
            "model_b": [10.0, 10.0, 3.0, 999.0, 1.0],
        }
    )
    original = frame.copy(deep=True)
    result = aggregate_rates(frame, "area", "loss", ["model_a", "model_b"], "exposure")
    np.testing.assert_allclose(result["loss"], [10, 4, 0])
    np.testing.assert_allclose(result["model_a"], [11, 5, 0])
    np.testing.assert_allclose(result["loss_total"], [40, 8, 0])
    np.testing.assert_allclose(result["model_a_total"], [44, 10, 0])
    np.testing.assert_allclose(result["model_a_difference"], [-1, -1, 0])
    np.testing.assert_allclose(result["model_a_ratio"], [10 / 11, 0.8, np.nan])
    assert result["support_count"].tolist() == [2, 1, 1]
    assert result["total_weight"].tolist() == [4, 2, 1]
    pd.testing.assert_frame_equal(frame, original)
    frame["loss"] = [10, 10, 4, 999, 0]
    rates = aggregate_rates(
        frame, "area", "loss", ["model_a", "model_b"], "exposure", observed_kind="rate"
    )
    pd.testing.assert_frame_equal(rates, result)


def test_zero_expected_is_undefined_even_with_nonzero_observed():
    frame = pd.DataFrame(
        {"id": [1, 1], "observed": [2.0, 4.0], "p": [0.0, 0.0], "e": [1.0, 3.0]}
    )
    result = aggregate_rates(frame, "id", "observed", ["p"], "e")
    assert result["id"].tolist() == [1]
    assert result.loc[0, "observed"] == 1.5
    assert result.loc[0, "p_difference"] == 1.5
    assert np.isnan(result.loc[0, "p_ratio"])


def test_derived_names_never_overwrite_metrics_or_geographic_ids():
    metrics = [
        "model",
        "model_difference",
        "model_ratio",
        "support_count",
        "total_weight",
    ]
    frame = pd.DataFrame(
        {
            "loss_total": ["001", "001"],
            "model_total": [10.0, 30.0],
            "exposure": [1.0, 3.0],
            **{name: [8.0, 12.0] for name in metrics},
        }
    )
    result = aggregate_rates(frame, "loss_total", "model_total", metrics, "exposure")
    assert result["loss_total"].tolist() == ["001"]
    assert result.loc[0, "model_total"] == 10
    for metric in metrics:
        assert result.loc[0, metric] == 11
        info = result.attrs["comparisons"][metric]
        assert result.loc[0, info["difference"]] == -1
        assert result.loc[0, info["ratio"]] == pytest.approx(10 / 11)
    assert result.columns.is_unique
    assert result.attrs["totals"]["model"] == "model_total_"


def test_validation_of_ids_schema_arguments_and_nullable_numbers():
    frame = pd.DataFrame(
        {"id": ["001", "001"], "v": pd.Series([10, 20], dtype="Float64")}
    )
    assert aggregate_means(frame, "id", ["v"]).loc[0, "v"] == 15
    with pytest.raises(ValueError, match="at least one"):
        aggregate_means(frame.iloc[:0], "id", ["v"])
    with pytest.raises(TypeError, match="metrics"):
        aggregate_means(frame, "id", "v")
    for metrics in ([], ["v", "v"]):
        with pytest.raises(ValueError, match="unique"):
            aggregate_means(frame, "id", metrics)
    with pytest.raises(ValueError, match="Missing columns.*absent"):
        aggregate_means(frame, "id", ["absent"])
    with pytest.raises(TypeError, match="weight"):
        aggregate_means(frame, "id", ["v"], [1, 1])
    with pytest.raises(ValueError, match="ID column"):
        aggregate_means(frame, "id", ["id"])
    with pytest.raises(ValueError, match="unique column"):
        aggregate_means(pd.concat([frame, frame[["v"]]], axis=1), "id", ["v"])
    frame.loc[0, "id"] = None
    with pytest.raises(ValueError, match="id.*missing IDs"):
        aggregate_means(frame, "id", ["v"])
    with pytest.raises(TypeError, match="real numeric"):
        aggregate_means(pd.DataFrame({"id": ["001"], "v": ["10"]}), "id", ["v"])
    with pytest.raises(ValueError, match="predicted"):
        aggregate_rates(frame, "id", "v", [], "e")
    with pytest.raises(ValueError, match="observed_kind"):
        aggregate_rates(frame, "id", "v", ["p"], "e", observed_kind="amount")


def test_finite_inputs_cannot_silently_overflow():
    frame = pd.DataFrame({"id": ["001", "001"], "v": [1e308, 1e308], "w": [1.0, 1.0]})
    with pytest.raises(ValueError, match="overflow"):
        aggregate_means(frame, "id", ["v"], "w")
    frame["w"] = [1e308, 1e308]
    with pytest.raises(ValueError, match="overflow"):
        aggregate_means(frame, "id", ["v"], "w")


def test_numerical_import_and_use_do_not_load_rendering():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import pandas as pd
from geomapviz import aggregate_means, aggregate_rates
df = pd.DataFrame({'id': ['001'], 'v': [10.], 'p': [8.], 'w': [2.]})
assert aggregate_means(df, 'id', ['v'], 'w').loc[0, 'v'] == 10
assert aggregate_rates(df, 'id', 'v', ['p'], 'w').loc[0, 'v'] == 5
for module in ['matplotlib', 'holoviews', 'geoviews', 'geopandas', 'seaborn']:
    assert module not in sys.modules, module
import geomapviz.aggregator as a
assert not hasattr(a, 'compute_confidence_interval')
""",
        ],
        check=True,
    )
