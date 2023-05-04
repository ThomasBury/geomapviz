# Settings and libraries
from __future__ import print_function
from os.path import dirname, join
from pkg_resources import resource_stream, resource_filename
# pandas
import pandas as pd
import geopandas as gpd

def load_geometry(shp_path: str, geoid: str = "INS") -> gpd.GeoDataFrame:
    """
    Load a shapefile and convert it to a GeoDataFrame with coordinates and projection set to Google Mercator.

    Parameters
    ----------
    shp_path : 
        The file path of the shapefile to load.
    geoid : 
        The name of the geoid column in the GeoDataFrame, by default "INS".

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame with the shapefile's geometry projected in Google Mercator and the geoid column cast to string.
    """
    geometries = gpd.read_file(shp_path)
    geom_merc = geometries.to_crs(ccrs.GOOGLE_MERCATOR.proj4_init)
    geom_merc[geoid] = geom_merc[geoid].astype(str)
    return geom_merc


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
    
    if country == "BE":
        data_filename = resource_filename(__name__, 'shp/belgium.shp')
    else:
        raise ValueError("The country must be one of ['BE']")
    
    return gpd.read_file(data_filename)
