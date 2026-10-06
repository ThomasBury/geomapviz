# Aggregation

Geomapviz returns inspectable pandas summaries without mutating input records.
Begin with [Belgian geographic means](examples/belgium.md),
[rate comparisons](examples/rates.md) or
[measured CBS percentages](examples/demographics.md). Each tutorial runs through
the canonical script and exports the inputs and summaries used in its maps.

## Means

`aggregate_means(records, geoid, metrics, weight=None)` returns one row per
observed ID and preserves metric names. With no weight column, every record has
weight one. With weights, each mean is `sum(value * weight) / sum(weight)`.
The Belgian example exports both equal and exposure-weighted results; its
unequal positive weights make the difference inspectable.

A compact arithmetic check:

```python
import pandas as pd
from geomapviz import aggregate_means

values = pd.DataFrame({"area": ["001", "001"], "value": [10, 20], "weight": [1, 3]})
means = aggregate_means(values, "area", ["value"], "weight")
assert means.loc[0, "value"] == 17.5
assert means.loc[0, "support_count"] == 2
assert means.loc[0, "total_weight"] == 4
```

Only observed categorical labels appear; unused categories do not become zero
areas. For percentages, choose the denominator deliberately. The
[CBS tutorial](examples/demographics.md) calculates ratios of published age and
resident counts, with one common valid cohort; it does not average age counts
or postcode percentages equally.

## Observed totals and rates

`aggregate_rates` requires observed values, predicted **rates** and exposure.
Exposure is the time or quantity at risk, such as insured years. The
[Belgian rates tutorial](examples/rates.md) uses simulated observed counts and
two deliberately biased predicted rates.

| Input or result | Computation within each area |
| --- | --- |
| Observed total, `observed_kind="total"` (default) | Sum observed amounts on positive-exposure rows once, then divide by exposure |
| Observed rate, `observed_kind="rate"` | Sum observed rate × exposure, then divide by exposure |
| Predicted rate | Sum predicted rate × exposure, then divide by exposure |
| Difference | Observed rate minus predicted rate |
| Ratio | Observed total divided by expected total |

The original metric names contain aggregate rates in both observed modes.
Derived columns contain observed/expected totals, differences and ratios. The
[rectangle reference](arithmetic.md) makes `40 / 4 = 10` and its comparison with
an exposure-weighted prediction easy to check by hand.

## Common cohorts and zero exposure

Choose a common comparison cohort before aggregation. All selected metrics and
weights must be real, numeric and finite, including on zero-weight records.
Missing values, infinity, negative weights, missing IDs, empty inputs, repeated
metrics and overflowing results fail explicitly. Different models never silently
receive different record subsets.

In the CBS example, negative suppression codes become missing, then one cohort
requires all selected counts and a positive resident denominator. Excluded
postcodes remain visible as missing geography. Exclusion is a caller decision;
it can change which population the result describes.

Zero-weight rows contribute neither numerators nor support counts. Every observed
area must have positive total weight: an area containing only zero-weight rows
fails. A boundary without records is instead retained as missing by
[geography preparation](geography.md#coverage).

## Undefined ratios

A zero expected total gives `NaN`, including `0 / 0`; never zero or infinity.
A difference may remain defined. The rates tutorial deliberately separates
missing Antwerpen, a genuine observed zero in Brussels and an undefined east
model ratio in Charleroi. Inspect its CSV and coverage JSON for their meanings.

## Support and derived names

`support_count` counts positive-weight records; `total_weight` sums weights
(exposure for rates). Equal means report record count in both columns.
These describe support, **not uncertainty estimates or confidence intervals**.
The CBS case has one input row per included postcode, so its count is one and
its total weight is the published resident denominator.

Caller metric names survive. Derived-name collisions append underscores until
a name is free. Read actual names from `summary.attrs["support"]`,
`summary.attrs["totals"]` and `summary.attrs["comparisons"]`.
`aggregate_rates` also records `attrs["observed_kind"]`. Preserve this metadata
through preparation and [plotting](plotting.md#metadata); CSV alone loses it.

At parent scale, [reaggregate original records](examples/aggregation.md).
Means and rates need their original numerators and denominators; do not average
child means or ratios. The downloadable checker verifies parent totals and weights.
