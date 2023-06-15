"""
Module for geographical visualization (geomapviz)
"""
# Settings and libraries
from __future__ import print_function
from os.path import dirname, join
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union, Tuple, List, Dict

# pandas
import pandas as pd
import geopandas as gpd
from mapclassify import FisherJenks
from dataclasses import dataclass

# numpy
import numpy as np
import math

# Plot and graphics
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import holoviews as hv
import geoviews as gv
from holoviews import opts
import contextily as cx

from .aggregator import dissolve_and_aggregate


hv.extension("bokeh", logo=False)
hv.renderer("bokeh").theme = "light_minimal"
sns.set(style="ticks")

__all__ = [
    "spatial_average_plot",
    "spatial_average_facetplot",
    "PlotOptions"
]

def dark_or_light_color(color: str):
    """Determine whether a given color is light or dark.

    Parameters
    ----------
    color :
        A color represented as a string in any valid matplotlib format.

    Returns
    -------
    str
        Either 'light' or 'dark', depending on whether the given color is
        light or dark.

    Raises
    ------
    ValueError
        If the given color is not a valid matplotlib color.

    Examples
    --------
    >>> dark_or_light_color('red')
    'dark'
    >>> dark_or_light_color('#FFFFFF')
    'light'
    >>> dark_or_light_color('invalid_color')
    Traceback (most recent call last):
        ...
    ValueError: Invalid RGBA argument: 'invalid_color'

    """
    [r, g, b] = mpl.colors.to_rgb(color)
    hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if hsp > 127.5:
        return "light"
    else:
        return "dark"


def set_decimal_precision(nbr_of_dec: str = 2, bins: Optional[np.ndarray] = None):
    """
    Set the decimal precision based on the number of decimal points in `nbr_of_dec`
    or based on the minimum absolute value in `bins` if `nbr_of_dec` is not provided.

    Parameters
    ----------
    nbr_of_dec :
        Number of decimal points to use.
    bins :
        Input array of bins.

    Returns
    -------
    str
        String representation of the number of decimal points to use.
    """

    if nbr_of_dec is not None:
        nbr_dec = str(nbr_of_dec)
    elif (bins is not None) and (np.abs(bins).min() < 1):
        nbr_dec = str(4)
    else:
        nbr_dec = str(2)

    return nbr_dec


def calculate_bins(
    df: pd.DataFrame,
    target: str = "avg",
    cols_to_bin: Optional[List[str]] = None,
    n_bins: int = 7,
    autobin: bool = False,
    normalize: bool = False,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Calculate the bin edges for each specified column in a given dataframe.

    Parameters
    ----------
    df :
        The input dataframe.
    target :
        The column to use as the target for binning (default is 'avg').
    cols_to_bin :
        The list of columns to calculate the bin edges for (default is None).
        If None, the bin edges will be calculated for all columns in the dataframe.
    n_bins :
        The number of bins to create (default is 7).
    autobin :
        Whether to automatically calculate the bin edges using Fisher-Jenks optimization (default is False).
    normalize :
        Whether to normalize the data before calculating the bin edges (default is False).

    Returns
    -------
    Dict[str, Optional[np.ndarray]]
        A dictionary mapping each column name to its corresponding bin edges.
        If autobin is False, the value for each column will be None.
    """
    if cols_to_bin is None:
        cols_to_bin = []
    
    bins_dict = {}
    for col in cols_to_bin:
        if autobin:
            bins = FisherJenks(df[target].fillna(0), n_bins).bins if normalize else (
                FisherJenks(df[col].fillna(0), n_bins).bins if df[col].nunique() > n_bins else None)
        else:
            bins = None
        bins_dict[col] = bins
    
    return bins_dict


def create_norm(
    df: pd.DataFrame, ref_col: str = "avg"
) -> Tuple[Normalize, float, float]:
    """
    Calculate the normalization for a color map based on the percentiles of a reference column.

    Parameters:
    -----------
    df:
        The data used to calculate the percentiles and normalization.
    ref_col:
        The name of the reference column in the dataframe.

    Returns:
    --------
    norm: mpl.colors.Normalize
        The normalization object for the color map.
    vmin: float
        The minimum value of the percentile range.
    vmax: float
        The maximum value of the percentile range.

    """
    # Calculate the 1st and 99th percentiles of the reference column
    vmin, vmax = np.nanpercentile(df[ref_col].fillna(0), [1, 99])
    # Create a normalization object using the percentile range
    norm = Normalize(vmin=vmin, vmax=vmax)
    return norm, vmin, vmax


def create_cbar(ax: plt.Axes, title_col: str):
    """
    Create a colorbar axis to the right of the given axis.

    Parameters
    ----------
    ax :
        The axis to add the colorbar to.
    title_col :
        The color of the colorbar title.

    Returns
    -------
    cax : plt.Axes
        The created colorbar axis.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.tick_params(
        axis="y",
        labelsize="large",
        labelcolor=title_col,
        grid_linewidth=0,
        color=title_col,
    )
    cax.set_frame_on(False)
    return cax


@dataclass
class PlotOptions:
    """Options for generating a thematic map using Matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data to plot.
    target : str
        The column name of the target variable to plot.
    other_cols_avg : Optional[str], optional
        The column name of other columns in the dataframe to be averaged and plotted against the target variable, by default None.
    weight : Optional[np.ndarray], optional
        An array of weights for each observation, by default None.
    plot_weight : bool, optional
        A boolean flag indicating whether to plot the weights on the map, by default False.
    dissolve_on : Optional[str], optional
        The column name of the column to dissolve on when aggregating geometries, by default None.
    geoid : str, optional
        The name of the column containing the geographic ID, by default "nis".
    shp_file : Optional[gpd.geodataframe.GeoDataFrame], optional
        A GeoDataFrame containing the geometry data to plot, by default None.
    distr : str, optional
        The distribution type to use when calculating bin thresholds for the target variable, by default "gaussian".
    plot_uncertainty : bool, optional
        A boolean flag indicating whether to plot uncertainty bands around the target variable, by default False.
    background : Optional[str], optional
        The name of the background map to use, by default None.
    figsize : Tuple[float, float], optional
        The figure size, by default (12, 12).
    ncols : int, optional
        The number of columns in the plot grid, by default 2.
    cmap : Optional[str], optional
        The name of the color map to use for the plot, by default None.
    facecolor : str, optional
        The background color of the plot, by default "#2b303b".
    nbr_of_dec : Optional[int], optional
        The number of decimal places to use when displaying values on the plot, by default None.
    autobin : bool, optional
        A boolean flag indicating whether to automatically calculate bin thresholds for the target variable, by default False.
    normalize : bool, optional
        A boolean flag indicating whether to normalize the color scale, by default True.
    n_bins : int, optional
        The number of bins to use when manually calculating bin thresholds for the target variable, by default 7.
    interactive : bool
        whether to use interactive charts or not.

    Returns
    -------
    PlotOptions
        A PlotOptions object containing the input arguments.
    """

    # data arguments
    df: pd.DataFrame
    target: str
    other_cols_avg: Optional[str] = None
    # weights arguments
    weight: Optional[np.ndarray] = None
    plot_weight: bool = False
    # geospatial arguments
    dissolve_on: Optional[str] = None
    geoid: str = "nis"
    shp_file: Optional[gpd.geodataframe.GeoDataFrame] = None
    # uncertainty arguments
    distr: str = "gaussian"
    plot_uncertainty: bool = False
    # style arguments
    alpha: float = 0.5
    background: Optional[str] = None
    figsize: Tuple[float, float] = (12, 12)
    ncols: int = 2
    cmap: Optional[str] = "plasma"
    facecolor: str = "#2b303b"
    nbr_of_dec: Optional[int] = None
    # binning arguments
    autobin: bool = False
    normalize: bool = True
    n_bins: int = 7
    # matplotlib or holoviews
    interactive = False


def plot_data(
    df: pd.DataFrame,
    bins_dict: Dict[str, Optional[List[float]]],
    ncols: int,
    facecolor: str,
    cmap: Optional[str],
    alpha: float,
    cols_to_enum: List[str],
    options: PlotOptions,
) -> mpl.figure.Figure:
    """
    Plots data for each column in a given DataFrame using the specified bins and other plotting options.

    Parameters
    ----------
    df :
        The input DataFrame containing the data to be plotted.
    bins_dict :
        A dictionary where keys are column names and values are lists of bin edges (if column is to be plotted using user-defined bins) or None (if column is to be plotted using equal interval binning).
    ncols :
        The number of columns to use for plotting the subplots.
    facecolor :
        The facecolor for the plots.
    cmap :
        The colormap to use for the plots. If None, use the default colormap.
    alpha :
        The alpha (transparency) value for the plotted data.
    cols_to_enum :
        The columns to plot.
    options :
        The options for plotting the data.

    Returns
    -------
    mpl.figure.Figure
        The resulting figure object containing the plotted data.
    """

    nrows = 1
    if ncols == 4:
        nrows, ncols = 2, 2

    f, axs = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=options.figsize, facecolor=options.facecolor
    )
    axs = axs.flatten() if ncols > 1 else axs

    luminance = dark_or_light_color(options.facecolor)
    title_col = "white" if luminance == "dark" else "black"

    if options.normalize:
        norm, _, _ = create_norm(df=df, ref_col="avg")

    # Plot each column in its corresponding axis
    for i, col in enumerate(cols_to_enum):
        ax = axs[i] if ncols > 1 else axs
        bins = bins_dict[col]
        if bins is None:
            _, vmin, vmax = create_norm(df=df, ref_col=col)
            cax = create_cbar(ax=ax, title_col=title_col)
            ax = df.plot(
                column=col,
                ax=ax,
                cax=cax,
                legend=True,
                linewidth=0,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                alpha=alpha,
            )
            if options.normalize:
                ax = df.plot(
                    column=col,
                    ax=ax,
                    cax=cax,
                    legend=True,
                    linewidth=0,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    alpha=alpha,
                    norm=norm,
                )
        else:
            nbr_dec = set_decimal_precision(options.nbr_of_dec, bins=bins)
            ax = df.plot(
                column=col,
                ax=ax,
                legend=True,
                linewidth=0,
                cmap=cmap,
                scheme="user_defined",
                alpha=alpha,
                classification_kwds={"bins": bins},
                legend_kwds={
                    "facecolor": facecolor,
                    "framealpha": 0,
                    "loc": "lower left",
                    "fmt": "{:." + nbr_dec + "f}",
                    "labelcolor": "#575757",

                },
            )
        if options.background:
            cx.add_basemap(ax, crs=df.crs, source=options.background)
        ax.set_axis_off()
        ax.set_title(
            "2 standard dev." if col == "2target_std" else col, color=title_col
        )

    return f


def spatial_average_plot(options: PlotOptions):
    """
    Plot data on a map using a GeoDataFrame.

    This function loads the data from a DataFrame `df`,
    aggregates it by geographic area, and plots the resulting averages
    on a map using a GeoDataFrame `shp_file`. The `target` column in `df` is
    used as the variable to plot on the map. Other
    columns can also be plotted using the `other_cols_avg` parameter.
    The `weight` parameter can be used to weight the data.
    The `dissolve_on` and `geoid` parameters are used to
    group the data by geographic area. The `autobin`, `normalize`, and `n_bins`
    parameters control the binning of the data. The `cmap` parameter controls the colormap,
    and the `facecolor` parameter controls the color of the plot background. The `plot_weight`
    and `plot_uncertainty` parameters control whether to plot the weight and uncertainty data,
    respectively. The resulting plot is returned as a matplotlib Figure object.

    Parameters
    ----------
    options : PlotOptions
        The options for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure object.

    Raises
    ------
    TypeError
        If `shp_file` in `options` is not a GeoDataFrame.
    """
    # Validate the shapefile
    if not isinstance(options.shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError("The shapefile should be a GeoDataFrame")

    # Load the data and dissolve
    geo_df = dissolve_and_aggregate(
        df=options.df,
        other_cols_avg=None,
        target=options.target,
        dissolve_on=options.dissolve_on,
        geoid=options.geoid,
        weight=options.weight,
        shp_file=options.shp_file,
        distr=options.distr,
    )

    # Define the colormap and the number of columns
    cmap = options.cmap or "plasma"
    alpha = 1.0 if options.background is None else 0.65
    ncols = 1 + 2 * options.plot_uncertainty + options.plot_weight

    # Define the columns to plot and the corresponding bins for each column
    cols_to_enum = ["avg"]
    weight_name = ["count"] if options.weight is None else ["weight"]

    if options.plot_uncertainty:
        cols_to_enum += ["ci_low", "ci_up"]
    if options.plot_weight:
        cols_to_enum += weight_name

    bins_dict = calculate_bins(
        df=geo_df,
        target="avg",
        autobin=options.autobin,
        cols_to_bin=cols_to_enum,
        n_bins=options.n_bins,
        normalize=options.normalize,
    )
    f = plot_data(
        df=geo_df,
        bins_dict=bins_dict,
        ncols=ncols,
        cols_to_enum=cols_to_enum,
        facecolor=options.facecolor,
        cmap=cmap,
        alpha=alpha,
        options=options,
    )
    return f


def calculate_bins_grouped_data(
    grouped: pd.core.groupby.DataFrameGroupBy, autobin: bool, n_bins: int, normalize: bool
):
    """
    Calculates the bin ranges for a given DataFrame of model averages.

    Parameters
    ----------
    grouped :
        The grouped data to plot. It should contain a "avg" column to plot as the main feature.
    autobin :
        Whether to use Fisher-Jenks algorithm to calculate bins based on the target model or not.
    n_bins :
        The number of bins to use when calculating bins with the Fisher-Jenks algorithm.
    normalize :
        Whether to use the same bin ranges for all models, or calculate them individually.

    Returns
    -------
    dict
        A dictionary containing the calculated bins for each model. Keys are model names, and values are lists
        of bin ranges. If a model's value is None, no binning was performed for that model.
    """
    bins_dict = {}
    target_df = grouped.get_group("target")
    ref_bins = (
        FisherJenks(target_df["avg"].fillna(0), n_bins)
        if autobin
        else None
    )
    for name, group in grouped:
        if autobin and normalize:
            bins_dict[name] = ref_bins
        elif autobin and (group["avg"].nunique() > n_bins):
            bins_dict[name] = FisherJenks(group["avg"].fillna(0), n_bins)
        else:
            bins_dict[name] = None
    return bins_dict


def plot_grouped_data(
    grouped: pd.core.groupby.DataFrameGroupBy,
    nrows: int,
    ncols: int,
    n_charts: int,
    bins_dict: dict,
    cmap: Union[str, mpl.colors.Colormap],
    alpha: float,
    title_col: str,
    nbr_of_dec: Optional[int] = None,
    facecolor: Optional[str] = None,
    background: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    normalize: bool = False,
):
    """Plot data grouped by a specified column, using matplotlib subplots.

    Parameters
    ----------
    grouped :
        The grouped data to plot. It should contain a "avg" column to plot as the main feature.
    nrows :
        The number of rows of subplots to create.
    ncols :
        The number of columns of subplots to create.
    n_charts :
        The number of charts to plot.
    bins_dict :
        A dictionary with the same keys as the groups in `grouped`, and values
        indicating the binning strategy to use for the corresponding group.
        If None, the data will be plotted without binning.
    cmap :
        The colormap to use for the plot.
    alpha :
        The alpha value to use for the plot.
    title_col :
        The color of the subplot titles.
    nbr_of_dec :
        The number of decimal places to use for the legend labels (default is None).
    facecolor :
        The hex facecolor of the figure (default is None).
    background : str or None, optional
        The background to use for the plot, as a string representing a basemap (default is None).
    figsize : tuple, optional
        The size of the figure, as a tuple of (width, height) in inches (default is (10,10)).

    Returns
    -------
    matplotlib.figure.Figure
        The resulting matplotlib figure.
    """

    f, axs = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=figsize, facecolor=facecolor
    )
    axs = axs.flatten() if ncols > 1 else axs

    target_df = grouped.get_group("target")
    
    if normalize:
        norm, vmin, vmax = create_norm(df=target_df, ref_col="avg")

    for i, (name, group) in enumerate(grouped):
        ax = axs[i] if ncols > 1 else axs
        bins = bins_dict[name]
        if bins is None:
            _, vmin, vmax = create_norm(df=group, ref_col="avg")
            # vmin, vmax = np.nanpercentile(group["avg"].fillna(0), [1, 99])
            cax = create_cbar(ax=ax, title_col=title_col)
            ax = group.plot(
                column="avg",
                ax=ax,
                cax=cax,
                legend=True,
                linewidth=0,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                alpha=alpha,
            )
            if normalize:
                ax = group.plot(
                    column="avg",
                    ax=ax,
                    cax=cax,
                    legend=True,
                    linewidth=0,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    alpha=alpha,
                    norm=norm,
                )
        else:
            nbr_dec = set_decimal_precision(nbr_of_dec, bins=bins.bins)
            group.plot(
                column="avg",
                ax=ax,
                legend=True,
                linewidth=0,
                cmap=cmap,
                scheme="user_defined",
                alpha=alpha,
                classification_kwds={"bins": bins.bins},
                legend_kwds={
                    "facecolor": facecolor,
                    "framealpha": 0,
                    "loc": "lower left",
                    "fmt": "{:." + nbr_dec + "f}",
                    "labelcolor": "#575757",
                },
            )
        if background:
            cx.add_basemap(ax, crs=group.crs, source=background)
        ax.set_axis_off()
        ax.set_title(name, color=title_col)

    # delete non-used axes
    n_subplots = nrows * ncols
    if n_subplots > n_charts > 1:
        for i in range(n_charts, n_subplots):
            ax = axs[i]
            ax.set_axis_off()
    return f


def spatial_average_facetplot(options: PlotOptions) -> mpl.figure.Figure:
    """
    Create a facet plot of spatial data on a map.

    Parameters
    ----------
    options : PlotOptions
        An object containing the options for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.

    Raises
    ------
    TypeError
        If the shapefile is not a GeoDataFrame.

    Notes
    -----
    This function uses the following helper functions: `dark_or_light_color()`, `dissolve_and_aggregate()`,
    `calculate_bins_grouped_data()`, and `plot_grouped_data()`.
    """
    # Validate the shapefile
    if not isinstance(options.shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError("The shapefile should be a GeoDataFrame")

    # Compute the luminance of the facecolor
    luminance = dark_or_light_color(options.facecolor)
    title_col = "white" if luminance == "dark" else "black"

    # Load the data and dissolve
    geo_df = dissolve_and_aggregate(
        df=options.df,
        target=options.target,
        other_cols_avg=options.other_cols_avg,
        dissolve_on=options.dissolve_on,
        geoid=options.geoid,
        weight=options.weight,
        shp_file=options.shp_file,
        distr=options.distr,
    )

    cols_to_plot = (
        [options.target]
        if options.other_cols_avg is None
        else [options.target] + options.other_cols_avg
    )
    ncols_to_plot = len(cols_to_plot)
    nrows = int(np.ceil(ncols_to_plot / options.ncols))

    # Fillna with 0 for the specified columns
    geo_df["model"] = geo_df["model"].fillna(0).values

    # Define the colormap
    cmap = options.cmap or "plasma"
    alpha = 1.0 if options.background is None else 0.65
    
    grouped = geo_df.groupby("model")

    bins_dict = calculate_bins_grouped_data(
        grouped=grouped,
        autobin=options.autobin,
        n_bins=options.n_bins,
        normalize=options.normalize,
    )
    if options.interactive:
        f = plot_grouped_data_interactive(
            grouped=grouped,
            ncols=options.ncols,
            bins_dict=bins_dict,
            cmap=options.cmap,
            alpha=options.alpha,
            nbr_of_dec=options.nbr_of_dec,
            background=options.background,
            figsize=options.figsize,
            normalize=options.normalize)
    else:
        f = plot_grouped_data(
            grouped=grouped,
            nrows=nrows,
            ncols=options.ncols,
            n_charts=ncols_to_plot,
            bins_dict=bins_dict,
            cmap=cmap,
            alpha=alpha,
            title_col=title_col,
            nbr_of_dec=options.nbr_of_dec,
            facecolor=options.facecolor,
            background=options.background,
            figsize=options.figsize,
            normalize=options.normalize,
        )

    return f


def get_tiles(tiles_src: Optional[str]) -> hv.element.tiles:
    """
    Get a HoloViews tiles element for a specified tiles source string.

    Parameters
    ----------
    tiles_src : str, optional
        The tiles source to use. If None, use "CartoLight" as the default.

    Returns
    -------
    hv.element.tiles
        The tiles element corresponding to the specified tiles source.
    """

    if tiles_src is None:
        tiles_src = "CartoLight"

    if tiles_src not in hv.element.tiles.tile_sources.keys():
        raise ValueError(
            f"`tiles_src` should be a string and one of {list(hv.element.tiles.tile_sources.keys())}"
        )

    return hv.element.tiles.tile_sources[tiles_src]()

def get_interactive_plot_options(col:str, cmap: Union[str, mpl.colors.Colormap], alpha:float):
    """
    Returns a dictionary of options for an interactive plot.

    Parameters
    ----------
    col : 
        The name of the column to plot.
    cmap : 
        The colormap to use for the plot.
    alpha :
        The opacity of the plot.

    Returns
    -------
    dict
        A dictionary of options for an interactive plot, including the following keys:
        - 'tools': list of str, specifying the tools to include in the plot.
        - 'width': int, specifying the width of the plot in pixels.
        - 'height': int, specifying the height of the plot in pixels.
        - 'color': geoviews.dim instance, specifying the column to use for coloring the plot.
        - 'cmap': str or matplotlib.colors.Colormap, specifying the colormap to use.
        - 'colorbar': bool, specifying whether to show a colorbar.
        - 'toolbar': str, specifying the location of the toolbar.
        - 'xaxis': None, to disable the x-axis.
        - 'yaxis': None, to disable the y-axis.
        - 'alpha': float, specifying the opacity of the plot.
        - 'title': str, specifying the title of the plot.
        - 'clipping_colors': dict, specifying colors to use for clipping values.

    """
    return dict(
            tools=["hover"],
            width=550,
            height=450,
            color=gv.dim("avg"),
            cmap=cmap,
            colorbar=True,
            toolbar="above",
            xaxis=None,
            yaxis=None,
            alpha=alpha,
            title=col,
            clipping_colors={"NaN": "white"},
        )

def get_facet(df: pd.DataFrame, tiles: hv.element.tiles, vmin: float, vmax: float, plot_opts: dict, 
              cbar_labels: Optional[list] = None) -> hv.core.overlay.Layout:
    """
    Create a choropleth map from a DataFrame of polygon geometries and their corresponding data.

    Parameters:
    -----------
    df : 
        A DataFrame containing polygon geometries and their corresponding data.
    tiles :
        A GeoViews element specifying the map tiles to use as the background.
    vmin : 
        The minimum value to use for the color bar.
    vmax : 
        The maximum value to use for the color bar.
    plot_opts :
        A dictionary of options to pass to the GeoViews element for styling the map.
    cbar_labels :
        A list of labels for the color bar. If provided, the color bar will be discrete with the 
        given labels as the major tick labels.

    Returns:
    --------
    facet : gv.core.overlay.Element
        A GeoViews element containing the choropleth map overlaid on the specified map tiles.
    """
    polygons = gv.Polygons(df, vdims=[hv.Dimension("avg", range=(vmin, vmax))],  
                                crs=ccrs.GOOGLE_MERCATOR).opts(**plot_opts)
    if cbar_labels:
        polygons.opts(colorbar_opts={'major_label_overrides': cbar_labels}, color_levels=len(cbar_labels))
    facet = tiles * polygons
    return facet

def plot_grouped_data_interactive(
    grouped: pd.core.groupby.DataFrameGroupBy,
    ncols: int,
    bins_dict: dict,
    cmap: Union[str, mpl.colors.Colormap],
    alpha: float,
    nbr_of_dec: Optional[int] = None,
    background: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    normalize: bool = False,
):
    """Plot data grouped by a specified column, using matplotlib subplots.

    Parameters
    ----------
    grouped :
        The grouped data to plot. It should contain a "avg" column to plot as the main feature.
    nrows :
        The number of rows of subplots to create.
    ncols :
        The number of columns of subplots to create.
    n_charts :
        The number of charts to plot.
    bins_dict :
        A dictionary with the same keys as the groups in `grouped`, and values
        indicating the binning strategy to use for the corresponding group.
        If None, the data will be plotted without binning.
    cmap :
        The colormap to use for the plot.
    alpha :
        The alpha value to use for the plot.
    title_col :
        The color of the subplot titles.
    nbr_of_dec :
        The number of decimal places to use for the legend labels (default is None).
    facecolor :
        The hex facecolor of the figure (default is None).
    background : str or None, optional
        The background to use for the plot, as a string representing a basemap (default is None).
    figsize : tuple, optional
        The size of the figure, as a tuple of (width, height) in inches (default is (10,10)).

    Returns
    -------
    matplotlib.figure.Figure
        The resulting matplotlib figure.
    """

    target_df = grouped.get_group("target")
    
    if normalize:
        norm, vmin, vmax = create_norm(df=target_df, ref_col="avg")

    background = get_tiles(tiles_src=background)
    facet_list = []
    
    for name, group in grouped:
        
        plot_opts = get_interactive_plot_options(col=name, cmap=cmap, alpha=alpha)

        bins = bins_dict[name]
        if bins is None:
            if not normalize:
                _, vmin, vmax = create_norm(df=group, ref_col="avg")
            facet = get_facet(df=group, tiles=background, vmin=vmin, vmax=vmax, plot_opts=plot_opts)
            facet_list.append(facet)

        else:
            dum = group.copy()
            dum["avg"] = dum["avg"].apply(lambda x: bins(x))
            # create a dictionary mapping values to labels
            label_dict = dict([(i, s) for i, s in enumerate(bins.get_legend_classes())])
            if nbr_of_dec is not None:
                dum["avg"] = dum["avg"].round(nbr_of_dec)
            # if not normalize:
            #     _, vmin, vmax = create_norm(df=dum, ref_col="avg")
            facet = get_facet(df=dum, tiles=background, vmin=None, vmax=None, plot_opts=plot_opts, cbar_labels=label_dict)
            facet_list.append(facet)
            
    hvl = (
        hv.Layout(facet_list)
        .opts(opts.Tiles(width=figsize[0], height=figsize[1]))
        .cols(ncols)
    )
    return hvl