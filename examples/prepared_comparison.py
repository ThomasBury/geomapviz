"""Prepared-summary workflow; all boundaries and records are synthetic.

Run: .venv/bin/python examples/prepared_comparison.py [--interactive]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from native_comparison import sample  # noqa: E402

from geomapviz import aggregate_rates, prepare_geography  # noqa: E402
from geomapviz.plot import PlotOptions, plot_geography  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/geomapviz-m4"))
    parser.add_argument(
        "--interactive", action="store_true", help="also export HTML maps"
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    records, boundaries = sample()
    summary = aggregate_rates(
        records, "area", "loss", ["model_a", "model_b"], "exposure"
    )
    mapped = prepare_geography(summary, boundaries, "area")
    metrics = ["loss", "model_a", "model_b", "model_a_difference", "model_a_ratio"]
    for autobin in (False, True):
        name = "classified" if autobin else "continuous"
        options = PlotOptions(autobin=autobin, ncols=3, figsize=(15, 10), alpha=0.8)
        figure = plot_geography(
            mapped, metrics, geoid="area", include_support=True, options=options
        )
        figure.savefig(output / f"{name}.png")
        plt.close(figure)
        if args.interactive:
            import holoviews as hv

            options.interactive = True
            layout = plot_geography(
                mapped, metrics, geoid="area", include_support=True, options=options
            )
            hv.save(
                layout, output / f"{name}.html", backend="bokeh", resources="inline"
            )
    print(summary.to_string(index=False))
    print(mapped.attrs["coverage"])
    print(f"Exported continuous/classified maps to {output}")


if __name__ == "__main__":
    main()
