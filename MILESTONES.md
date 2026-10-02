# Geomapviz 2.0 implementation evidence

## M0 — complete (2026-10-02)

Created local branch `feat/v2.0.0` from `main` at
`3d1d45e31809135f7cab8d76e1e38838385ba1b2` before package changes.
The base version is `1.1.3`; main and the 1.x history remain available.
The only initial working-tree change was untracked `PRD.md`, preserved as supplied.
No push or publication is part of this work.

Environment: Linux x86_64, glibc 2.39, CPython 3.12.7, uv 0.12.15.
Installed `.[test,lint]` and hvPlot in a local `.venv` for the native baseline.
Resolved versions: pandas 3.0.6, NumPy 2.5.3, GeoPandas 1.2.0,
Matplotlib 3.11.2, HoloViews 1.23.2, GeoViews 1.15.1, hvPlot 0.12.2.

Baseline results, before repairs:

- `python -m pytest -q`: collection fails because `geomapviz.geomapviz` is absent.
- `python -m black --check src tests`: existing plot and shapefile modules fail.
- `python -m flake8 src tests`: existing long lines, invalid docstring escapes,
  and an unused pytest import fail. No repository flake8 configuration exists.
- The weighted mean of `[10, 20]` with weights `[1, 3]` is 17.5, but the
  metric is renamed to `target`. `[10, missing]` with the same weights is
  silently accepted and returns 2.5.
- `import geomapviz.plot` succeeds with the complete installed stack.

These are baseline defects, not regressions from M2. Rendering repairs,
dependency cleanup, and distribution verification belong to M3–M6.
