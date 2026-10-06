"""Validated pandas summaries; rendering is not required."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import check_list_of_str

__all__ = ["aggregate_means", "aggregate_rates"]


def _validate(df, geoid, metrics, weight):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df must contain at least one observation")
    if not isinstance(geoid, str):
        raise TypeError("geoid must be a column name")
    check_list_of_str(metrics, "metrics")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a non-empty list of unique column names")
    if weight is not None and not isinstance(weight, str):
        raise TypeError("weight must be a column name or None")
    if geoid in metrics or geoid == weight:
        raise ValueError(f"Geographic ID column {geoid!r} cannot be a metric or weight")
    if not df.columns.is_unique:
        raise ValueError("df must have unique column names")
    selected = list(
        dict.fromkeys([geoid] + metrics + ([weight] if weight is not None else []))
    )
    missing = [name for name in selected if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[geoid].isna().any():
        raise ValueError(f"Geographic ID column {geoid!r} contains missing IDs")
    numbers = {}
    for name in selected[1:]:
        series = df[name]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_complex_dtype(
            series
        ):
            raise TypeError(f"Column {name!r} must contain real numeric values")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"Column {name!r} contains missing or non-finite values")
        numbers[name] = values
    values = pd.DataFrame({name: numbers[name] for name in metrics}, index=df.index)
    weights = pd.Series(numbers[weight] if weight is not None else 1.0, index=df.index)
    if (weights < 0).any():
        raise ValueError(f"Weight column {weight!r} contains negative values")
    return values, weights


def _totals(df, geoid, metrics, weight, total_metric=None):
    values, weights = _validate(df, geoid, metrics, weight)
    grouping = dict(observed=True, sort=False)
    total_weight = weights.groupby(df[geoid], **grouping).sum()
    unsupported = total_weight.index[total_weight <= 0].tolist()
    if unsupported:
        raise ValueError(f"Areas with no positive total weight: {unsupported}")
    if not np.isfinite(total_weight.to_numpy()).all():
        raise ValueError(f"Weight column {weight!r} overflows during aggregation")
    counts = (weights > 0).groupby(df[geoid], **grouping).sum()
    with np.errstate(over="ignore", invalid="ignore"):
        numerators = values.mul(weights, axis=0)
    if total_metric is not None:
        numerators[total_metric] = values[total_metric].where(weights > 0, 0)
    if not np.isfinite(numerators.to_numpy()).all():
        raise ValueError("Weighted metric values overflow during aggregation")
    totals = numerators.groupby(df[geoid], **grouping).sum()
    if not np.isfinite(totals.to_numpy()).all():
        raise ValueError("Metric totals overflow during aggregation")
    return totals, total_weight, counts


def _add_column(result, name, values):
    """Append underscores until a derived name cannot overwrite caller columns."""
    while name in result.columns:
        name += "_"
    result[name] = values
    return name


def _summary(totals, total_weight, counts):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        means = totals.div(total_weight, axis=0)
    if not np.isfinite(means.to_numpy()).all():
        raise ValueError("Aggregate rates or means overflow")
    result = means.reset_index()
    result.attrs["support"] = {
        "count": _add_column(result, "support_count", counts.to_numpy()),
        "weight": _add_column(result, "total_weight", total_weight.to_numpy()),
    }
    return result


def aggregate_means(
    df: pd.DataFrame,
    geoid: str,
    metrics: list[str],
    weight: str | None = None,
) -> pd.DataFrame:
    """Return named means by geographic ID, using one common finite cohort.

    Weights must be finite and non-negative. Zero weights contribute neither
    to the numerator nor the support count; every observed area must have
    positive total weight. Without a weight column, all records have weight 1.
    Missing/non-finite metrics are rejected even on zero-weight records:
    callers must select their comparison cohort explicitly.

    IDs (including categorical labels) and metric names are preserved. Support
    columns start at ``support_count`` and ``total_weight``; collisions append
    underscores. ``result.attrs['support']`` maps count/weight to actual names.
    No statistical intervals are computed. Caller data is never changed.
    """
    return _summary(*_totals(df, geoid, metrics, weight))


def aggregate_rates(
    df: pd.DataFrame,
    geoid: str,
    observed: str,
    predicted: list[str],
    exposure: str,
    *,
    observed_kind: str = "total",
) -> pd.DataFrame:
    """Compare observed totals or rates with exposure-weighted predicted rates.

    ``observed_kind='total'`` sums observed amounts once and divides by summed
    exposure. ``'rate'`` weights existing observed rates by exposure, just like
    predictions. Zero-exposure rows are excluded from totals and support.
    The original metric names contain aggregate rates in the returned frame.

    Additional columns contain observed/expected totals, signed observed minus
    predicted rate differences, and observed/expected total ratios. An expected
    total of zero yields NaN. Derived names append underscores on collision;
    ``attrs['totals']`` maps each metric to its total column and
    ``attrs['comparisons']`` maps each prediction to difference/ratio columns.
    Support and validation follow ``aggregate_means``; exposure is required.
    """
    if not isinstance(observed, str) or not isinstance(exposure, str):
        raise TypeError("observed and exposure must be column names")
    check_list_of_str(predicted, "predicted")
    if not predicted:
        raise ValueError("predicted must be a non-empty list of rate column names")
    if observed_kind not in ("total", "rate"):
        raise ValueError("observed_kind must be 'total' or 'rate'")
    metrics = [observed] + predicted
    if exposure in metrics:
        raise ValueError("exposure must be distinct from observed/predicted metrics")
    totals, total_weight, counts = _totals(
        df, geoid, metrics, exposure, observed if observed_kind == "total" else None
    )
    result = _summary(totals, total_weight, counts)
    result.attrs["observed_kind"] = observed_kind
    result.attrs["totals"] = {
        metric: _add_column(result, f"{metric}_total", totals[metric].to_numpy())
        for metric in metrics
    }
    comparisons = {}
    for model in predicted:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            difference = result[observed] - result[model]
            ratio = totals[observed] / totals[model].where(totals[model] != 0)
        if (
            not np.isfinite(difference.to_numpy()).all()
            or np.isinf(ratio.to_numpy()).any()
        ):
            raise ValueError(f"Comparison with {model!r} overflows")
        comparisons[model] = {
            "difference": _add_column(result, f"{model}_difference", difference),
            "ratio": _add_column(result, f"{model}_ratio", ratio.to_numpy()),
        }
    result.attrs["comparisons"] = comparisons
    return result
