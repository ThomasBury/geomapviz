"""Validated joins to caller-supplied boundaries and parent mappings."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd


def _ids(frame, column, label):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a DataFrame")
    if not frame.columns.is_unique:
        raise ValueError(f"{label} must have unique column names")
    if not isinstance(column, str):
        raise TypeError("Geographic ID must be a column name")
    if column not in frame.columns:
        raise ValueError(f"{label} is missing ID column {column!r}")
    if frame.empty:
        raise ValueError(f"{label} must contain at least one area or observation")
    if frame[column].isna().any():
        raise ValueError(f"{label} column {column!r} contains missing IDs")

    def normalize(value):
        if isinstance(value, str):
            return value
        if isinstance(value, Integral):
            return str(value)
        if isinstance(value, Real) and np.isfinite(value):
            return str(int(value)) if value == int(value) else str(value)
        raise ValueError(f"{label} column {column!r} has invalid ID {value!r}")

    return frame[column].astype(object).map(normalize).astype("string")


def _boundaries(boundaries, geoid):
    import geopandas as gpd

    if not isinstance(boundaries, gpd.GeoDataFrame):
        raise TypeError("boundaries must be a GeoDataFrame")
    ids = _ids(boundaries, geoid, "boundaries")
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(
            f"Duplicate boundary IDs; dissolve explicitly first: {duplicates}"
        )
    if boundaries.active_geometry_name is None:
        raise ValueError("boundaries must have an active geometry column")
    if boundaries.geometry.name == geoid:
        raise ValueError("Geographic ID cannot be the active geometry column")
    if boundaries.crs is None:
        raise ValueError("Boundary CRS is required")
    geometry = boundaries.geometry
    invalid = geometry.isna() | geometry.is_empty | ~geometry.is_valid
    if invalid.any():
        raise ValueError(
            f"Missing, empty or invalid boundary geometries: {ids[invalid].tolist()}"
        )
    return ids


def _check_coverage(ids, boundary_ids):
    unknown = ids[~ids.isin(boundary_ids)].unique().tolist()
    if unknown:
        raise ValueError(f"Observation IDs without boundaries: {unknown}")


def prepare_geography(
    summary: pd.DataFrame, boundaries: gpd.GeoDataFrame, geoid: str
) -> gpd.GeoDataFrame:
    """Left join one summary per area, retaining every boundary and its CRS.

    IDs become strings on both sides: categorical labels and leading zeros
    survive; integral numeric IDs such as 1 and 1.0 match '1', never '001'.
    Unknown observation IDs and ambiguous boundaries fail explicitly. Boundary
    attributes that collide with summary columns are rejected; select the
    desired boundary columns first. Inputs are unchanged.

    Summary metadata is retained, with ``attrs['coverage']`` containing lists
    of matched and unmatched boundary IDs. Missing areas retain NaN metrics.
    Use native ``to_crs`` when a renderer needs another projection.
    """
    boundary_ids = _boundaries(boundaries, geoid)
    ids = _ids(summary, geoid, "summary")
    duplicates = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(f"Duplicate summary IDs: {duplicates}")
    _check_coverage(ids, boundary_ids)
    collisions = [
        name for name in summary.columns if name != geoid and name in boundaries.columns
    ]
    if collisions:
        raise ValueError(f"Boundary and summary columns overlap: {collisions}")
    geometry, values = boundaries.copy(), summary.copy()
    geometry[geoid] = boundary_ids
    values[geoid] = ids
    mapped = geometry.merge(values, on=geoid, how="left", validate="one_to_one")
    mapped.attrs = summary.attrs.copy()
    matched = boundary_ids.isin(ids)
    mapped.attrs["coverage"] = {
        "matched": boundary_ids[matched].tolist(),
        "unmatched": boundary_ids[~matched].tolist(),
    }
    return mapped


def assign_parent(
    records: pd.DataFrame, boundaries: gpd.GeoDataFrame, geoid: str, parent: str
) -> pd.DataFrame:
    """Copy records and assign normalized parent IDs from validated boundaries.

    Require one boundary and one non-missing parent per base ID. Existing
    record parent labels must agree with this mapping. Original base IDs,
    row order, index and metrics are retained. Aggregate the returned records
    with ``aggregate_means`` or ``aggregate_rates`` using ``geoid=parent``;
    never average base-area averages. Dissolve the boundary geometry with
    native GeoPandas separately, then call ``prepare_geography``.
    """
    boundary_ids = _boundaries(boundaries, geoid)
    if parent == geoid or parent == boundaries.geometry.name:
        raise ValueError("parent must be distinct from the base ID and geometry")
    parents = _ids(boundaries, parent, "boundaries")
    ids = _ids(records, geoid, "records")
    _check_coverage(ids, boundary_ids)
    mapping = pd.Series(parents.to_numpy(), index=boundary_ids)
    assigned = ids.map(mapping).astype("string")
    if parent in records.columns:
        existing = _ids(records, parent, "records")
        conflicts = ids[existing != assigned].unique().tolist()
        if conflicts:
            raise ValueError(
                f"Record parent labels disagree with boundaries: {conflicts}"
            )
    result = records.copy()
    result[parent] = assigned
    return result


def load_geometry(shp_path: str, geoid: str = "INS") -> gpd.GeoDataFrame:
    """
    Load a file while preserving its declared CRS and geographic ID labels.

    Parameters
    ----------
    shp_path :
        The file path of the shapefile to load.
    geoid :
        The name of the geoid column in the GeoDataFrame, by default "INS".

    Returns
    -------
    gpd.GeoDataFrame
        Validated boundaries in the file's declared CRS, with normalized IDs.
    """
    import geopandas as gpd

    geometries = gpd.read_file(shp_path)
    geometries[geoid] = _boundaries(geometries, geoid)
    return geometries


def load_shp(country: str = "BE"):
    """
    Load a shapefile of a specific country.

    Parameters
    ----------
    country :
        The ISO 3166-1 alpha-2 code of the country to load. Default is "BE" for Belgium.

    Returns
    -------
    gpd.geodataframe.GeoDataFrame
        A GeoDataFrame containing the shapefile data of the specified country.

    Raises
    ------
    ValueError
        If the specified country code is invalid or the corresponding shapefile is not found.

    Examples
    --------
    >>> belgium = load_shp("BE")
    >>> belgium.plot()
    """

    import geopandas as gpd
    from pkg_resources import resource_filename

    shp_files = {
        "BE": "belgium.shp",
        "NL": "nl_pc4_2014.shp",
        "IT": "italy_region.shp",
    }

    if country in shp_files:
        data_filename = resource_filename(__name__, f"shp/{shp_files[country]}")
    else:
        raise ValueError("The country must be one of ['BE', 'NL', 'IT']")

    return gpd.read_file(data_filename)
