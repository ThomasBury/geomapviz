# Aggregation

**2.0 development documentation.** All aggregation functions return pandas
DataFrames and leave the original records unchanged. They do not import the
rendering stack or compute statistical intervals.

## Means

`aggregate_means(records, geoid, metrics, weight=None)` preserves metric names
and returns one row per observed geographic ID. With no weight column, each
record has weight one. With weights, each mean is `sum(value × weight) / sum(weight)`.

```python
import pandas as pd
from geomapviz import aggregate_means

values = pd.DataFrame({"area": ["001", "001"], "value": [10, 20], "weight": [1, 3]})
means = aggregate_means(values, "area", ["value"], "weight")
assert means.loc[0, "value"] == 17.5
assert means.loc[0, "support_count"] == 2
assert means.loc[0, "total_weight"] == 4
```

Only observed categorical labels appear in the summary; unused categories
do not become artificial zero areas.

## Observed totals and rates

Use `aggregate_rates` for exposure-based comparisons. Exposure is the amount
of support, such as insured years. Predictions must be rates per unit of exposure.

| Input | Computation within each area |
| --- | --- |
| Observed amount, `observed_kind="total"` (default) | Sum amounts on positive-exposure rows once, then divide by total exposure |
| Observed rate, `observed_kind="rate"` | Sum observed rate × exposure, then divide by total exposure |
| Predicted rate | Sum predicted rate × exposure, then divide by total exposure |
| Difference | Observed rate minus predicted rate |
| Ratio | Observed total divided by expected total |

The [Quickstart](index.md#a-complete-synthetic-comparison) uses observed amounts.
To represent the same outcomes as input rates, divide only positive-exposure
rows; choose an explicit finite value for excluded rows:

```python
# Continue from the Quickstart's records and summary.
from geomapviz import aggregate_rates

rate_records = records.copy()
positive = rate_records["exposure"] > 0
rate_records["loss"] = 0.0
rate_records.loc[positive, "loss"] = (
    records.loc[positive, "loss"] / records.loc[positive, "exposure"]
)
rate_summary = aggregate_rates(
    rate_records, "area", "loss", ["model_a", "model_b"], "exposure",
    observed_kind="rate",
)
pd.testing.assert_frame_equal(rate_summary, summary)
```

Original metric columns contain aggregate rates in both modes. Additional
columns normally have names such as `loss_total`, `model_a_total`,
`model_a_difference` and `model_a_ratio`.

## Common cohorts and zero exposure

Select one comparison cohort before calling either function. Every selected
metric and weight must be real, numeric and finite. Missing values and infinity
are rejected, including on zero-weight records. Negative weights, missing IDs,
empty inputs, repeated metric names and overflowing aggregates also fail.
The functions do not silently drop different rows for different models.

For example, if missing predictions justify excluding records, apply the same
finite-row rule to all selected metrics and exposure, and inspect the exclusions:

```python
import numpy as np

columns = ["loss", "model_a", "model_b", "exposure"]
finite = np.isfinite(records[columns].to_numpy(dtype=float)).all(axis=1)
cohort = records.loc[finite].copy()
excluded_count = int((~finite).sum())
summary = aggregate_rates(cohort, "area", "loss", ["model_a", "model_b"], "exposure")
```

This selects finite records; deciding whether their exclusion is appropriate
remains the caller's responsibility. It does not fix negative exposure or missing IDs.

Zero-weight rows contribute neither numerators nor support counts. Every observed
area must still have positive total weight; an area represented only by
zero-exposure rows raises an error. That differs from a boundary with no records,
which [geography preparation](geography.md#coverage) retains as missing.

## Undefined ratios

An expected total of zero gives `NaN`, including `0 / 0`. It is never replaced
with zero or infinity. A signed difference can still be defined when the ratio
is not. Map missing/undefined ratios as such rather than filling them with zero.

## Support and derived names

`support_count` counts positive-weight records; `total_weight` is the sum of
weights (exposure for rates). Unweighted means report the record count in both.
These describe support, not confidence intervals.

Caller metric names are preserved. Derived-name collisions append underscores
until the name is free. Read the actual names from summary metadata:

```python
count_column = summary.attrs["support"]["count"]
exposure_column = summary.attrs["support"]["weight"]
total_column = summary.attrs["totals"]["loss"]
ratio_column = summary.attrs["comparisons"]["model_a"]["ratio"]
```

`aggregate_rates` also records `attrs["observed_kind"]`. Preserve this metadata
through [geography preparation](geography.md) and [plotting](plotting.md#metadata)
so totals, rates, differences, ratios and support use appropriate scales.
