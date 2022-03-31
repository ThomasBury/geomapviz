"""
Module for geographical visualization (geomapviz)
"""
# Settings and libraries
from __future__ import print_function
from os.path import dirname, join
from mpl_toolkits.axes_grid1 import make_axes_locatable
from palettable.cartocolors.qualitative import Bold_10
from pkg_resources import resource_stream, resource_filename
# pandas
import pandas as pd
import geopandas as gpd
from mapclassify import Quantiles
import warnings

# numpy
import numpy as np

# Plot and graphics
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr
import cartopy.crs as ccrs
import holoviews as hv
import geoviews as gv
from holoviews import opts

hv.extension("bokeh", logo=False)
hv.renderer("bokeh").theme = "light_minimal"
sns.set(style="ticks")

__all__ = [
    "single_predictor_aggr_multi_mod",
    "merge_zip_df",
    "prepare_geo_data",
    "plot_on_map",
    "facet_map",
    "facet_map_interactive",
    "load_be_shp",
]


def single_predictor_aggr_multi_mod(
    df,
    feature,
    target,
    weight,
    predicted,
    distrib="gaussian",
    ci=True,
    autobin=False,
    n_vals=50,
    n_bins=20,
    names=None,
    verbose=False,
):
    """
    Data aggregator to compute the uni-variate target for multiple models.
    Group by [feature] and sum [weight, target, predicted[model_i]].
    The aggregated target is then obtained by dividing the sum(target)/sum(weight) or
    sum(predicted)/sum(weight).
    If you don't want a weighted average, set weight as None,
    it'll pass a vector of ones as weight.

    The feature chosen to group by can be auto-binned. If the number of unique values
    df[feature].nunique() > n_vals then n_bins are built using a quantile cut.
    This is done to avoid to bin a predictor with already few unique values.
    The number of minimum unique values (n_vals) and the number of bins (n_bins)
    can be chosen at will.

    Intended to compute the uni-variate claim freq or sev,
    the confidence intervals are derived at the same time, using
    the assumption that the ncl is Poisson and sev is Gamma distributed.
    You can disable the computation.


    :param df: dataframe
        the data set, predictors + dependent variable
    :param feature: str
        the predictor to group by
    :param target: str
        the target (dependent variable), observed column
    :param weight: str or None
        the name of the column weight or None. If none,
        the weights are a vector of ones
    :param predicted: series or array
        the predicted values
    :param distrib: str, default="gaussian"
        the kind of distrib "poisson" or "gamma" or other (Gaussian),
        to compute the confidence intervals accordingly
    :param ci: bool, default=True
        compute or not the confidence intervals.
        Set distrib to 'gaussian' for the Gaussian approximation
    :param autobin: bool, default=False
        autobin or not the predictor to group by
    :param n_vals: int, default=50
        the minimal number of unique values that the predictor
        should have to be binned, because it makes little sense
        to bin a predictor with few unique values.
    :param n_bins: int, default=20
        the number of bins if autobin is set to true
    :param names: list, default None
        the list of strings, columns names, in the output dataframe
    :param verbose: Bool, default False
        print or not messages
    :return:
     df_average: dataframe
        the df, in long format, with the weighted average and
        the confidence intervals (if computed)
     df_: dataframe
        the df, in wide format, with only the weighted average
    """
    # sanity checks
    if not isinstance(feature, str):
        raise TypeError("`feature` should be a string")

    if not isinstance(target, str):
        raise TypeError("`target` must be a string ")

    if weight is not None:
        if not isinstance(weight, str):
            raise TypeError("`weight` should be a string")

    # if weight not provided, then set to one -> no effect
    if weight is None:
        weight = "weight"
        df_ = df[[feature, target]].copy()
        df_[weight] = 1
        if verbose:
            print(
                "Weight not provided, using the Gaussian approx for the "
                "CI"
            )
    else:
        df_ = df[[feature, target, weight]].copy()
    # to count rows for each level
    df_["count"] = 1

    # use the categorical codes, if any (it avoids issues)
    cat_cols = df_.select_dtypes("category").columns.tolist()
    if cat_cols:
        for c in cat_cols:
            df_[c] = df_[c].cat.codes

    # if other columns than the target to aggregate, performs sanity checks
    # set columns names (if any) and concatenate to the main dataframe
    if predicted is not None:
        col_y = None
        if isinstance(predicted, pd.DataFrame):
            col_y = predicted.columns.tolist()
        elif isinstance(predicted, pd.Series):
            col_y = [predicted.name]
        elif isinstance(predicted, list) and all(isinstance(s, str) for s in predicted):
            col_y = predicted
        else:
            predicted = pd.DataFrame(predicted)

        # setting the names
        if isinstance(predicted, pd.DataFrame):
            if names:
                error_message = (
                    "names should be a list (of strings) the same "
                    "length of the predicted dataFrame"
                )
                assert len(names) == len(predicted.columns), error_message
                predicted.columns = names
            else:
                predicted = predicted.add_prefix("model_")
            col_y = predicted.columns.tolist()
            df_ = pd.concat(
                [df_.reset_index(drop=True), predicted.reset_index(drop=True)], axis=1
            )
        elif isinstance(predicted, pd.Series):
            if names:
                error_message = (
                    "names should be a string (a single column in predicted)"
                )
                assert isinstance(names, str), error_message
                predicted.name = names
            col_y = predicted.columns.tolist()
            df_ = pd.concat(
                [df_.reset_index(drop=True), predicted.reset_index(drop=True)], axis=1
            )
        elif isinstance(predicted, list) and all(isinstance(s, str) for s in predicted):
            df_ = pd.concat(
                [df_.reset_index(drop=True), df[predicted].reset_index(drop=True)],
                axis=1,
            )

        pred_obs_cols = col_y + [target]

    else:
        pred_obs_cols = [target]

    # weighted average is not at the geomid level.
    # In order to compute the average at the geomid level, we need
    # to get back the original values
    # e.g. if the target is a rate (wealth / person) then we need
    # to get back the quantity,
    # group by sum them at the geomid level and then divide the
    # result to have the rate at the geomid level
    for col in pred_obs_cols:
        df_[col] = df_[col] * df_[weight]

    # autobin is obtained using the quantile cut
    if autobin:
        if df_[feature].nunique() > n_vals:
            df_[feature] = (
                pd.qcut(df_[feature], q=n_bins, duplicates="drop")
                .apply(lambda x: x.mid.round(0))
                .astype(float)
            )

    # summing the target and weight
    df_ = df_.groupby([feature])[pred_obs_cols + [weight, "count"]].sum().reset_index()

    # get the rate at the geomid level
    df_.loc[:, pred_obs_cols] = df_.loc[:, pred_obs_cols].div(
        df_[weight].values, axis=0
    )
    if distrib == "poisson":
        df_["target_std"] = np.sqrt(df_[target] / df_[weight])
    elif distrib == "gamma":
        df_["target_std"] = df_[target] * np.sqrt(1 / df_[weight])
    else:
        df_["target_std"] = df_[target] * np.sqrt(1 / df_["count"])

    df_average = pd.melt(
        df_[[feature, weight, "count"] + pred_obs_cols].copy(),
        id_vars=[feature, weight, "count"],
        var_name="model",
        value_name="target",
    )

    # Compute the confidence intervals
    if ci:
        if distrib == "poisson":
            print("Poisson CI")
            df_average["target_std"] = np.sqrt(
                df_average["target"] / df_average[weight]
            )
        elif distrib == "gamma":
            print("Gamma CI")
            df_average["target_std"] = df_average["target"] * np.sqrt(
                1 / df_average[weight]
            )
        else:
            print("Gaussian CI")
            df_average["target_std"] = df_average["target"] * np.sqrt(
                1 / df_average["count"]
            )

        df_average["ci_low"] = df_average["target"] - 2 * df_average["target_std"]
        df_average["ci_low"] = df_average["ci_low"].clip(lower=0)
        df_average["ci_up"] = df_average["target"] + 2 * df_average["target_std"]
        upper_bound = df_average["ci_up"].quantile(0.975)
        df_average["ci_up"] = df_average["ci_up"].clip(upper=upper_bound)
        df_average = df_average.reset_index()[
            ["model", feature, "target", "ci_low", "ci_up", "target_std"]
        ]

    return df_average, df_


def merge_zip_df(zip_path, df, geoid="geoid", cols_to_keep=None):
    """
    Merge the data (df) with a mapping table for the zipcode and other relevant
    geographical information (district name, sub-districts etc.). The key is the geoid.
    The zip mapper could be something like:

    +----+---------+-------------+---------+---------+------------+-----------+-------------------+
    |    |   geoid | town        |     lat |    long |   postcode | district  | borough           |
    +====+=========+=============+=========+=========+============+===========+===================+
    |  0 |   21004 | BRUSSEL     | 50.8333 | 4.35    |       1000 | Brussels  | Brussel Hoofdstad |
    +----+---------+-------------+---------+---------+------------+-----------+-------------------+
    |  1 |   21015 | SCHAARBEEK  | 50.85   | 4.38333 |       1030 | Brussels  | Brussel Hoofdstad |


    :param zip_path: str
        the path to the zipcode mapper, a csv file with additional
        geo info and a geoid column
    :param df: pd.DataFrame
        the data set
    :param geoid: str, default='geoid'
        the geoid column name, for merging with additional geographical information
    :param cols_to_keep: list of str, default=None
        the list of columns to keep
    :return: df, pa.DataFrame
        the data merged with geo info


    """
    if (not isinstance(cols_to_keep, list)) and (cols_to_keep is not None):
        raise TypeError("If `cols_to_keep` is not None, it should be a list of strings")

    # load zipcode mapper and adding borough to the dataframe
    zip_df = pd.read_csv(zip_path)
    zip_df["geoid"] = zip_df["geoid"].astype(str)
    if cols_to_keep is not None:
        zip_map = zip_df[["geoid"] + cols_to_keep].copy()
    else:
        zip_map = zip_df.copy()
    df[geoid] = df[geoid].astype(str)
    zip_map["geoid"] = zip_map["geoid"].astype(str)
    df = pd.merge(df, zip_map, how="left", left_on=[geoid], right_on=["geoid"])
    if geoid != "geoid":
        df = df.drop([geoid], axis=1)
    return df


def prepare_geo_data(
    df,
    cols_to_plot,
    target,
    predicted=None,
    dissolve_on=None,
    distrib="gaussian",
    n_bins=7,
    geoid="INS",
    weight=None,
    shp_file=None,
    autobin=False
):
    """
    Prepare the geodata to map. Take the dataframe df, which should have a
    geoid column, and join it to the zipcode mapper and the geometries.

    An aggregation step by geoid is performed in order to have a
    value per unique geoid.

    If weight is provided, then the target (and predicted if not None)
    are multiplied: target*weight, in order to compute the correct average
    value for each different geoid.

    If dissolve it set to True, the geometries are "dissolved" to go
    at the upper level, e.g. dissolve counties to go at the state level
    (a state being larger and containing several counties).


    :param df: pd.DataFrame
        the data set, predictors + dependent variable
    :param cols_to_plot: list of str of None
        the columns to plot with the target (e.g. predicted values or
        any other columns than target)
    :param target: str
        the target name (main column to plot, e.g. observed values/truth)
    :param predicted: str or pandas data frame or None, default: None
        if dataframe, should be the same length of df.
        It could be predicted values, not joined to df
    :param dissolve_on: str
        the column name you want to dissolve on.
        Dissolve means going to an upper geographical level,
        e.g. from commune to district or to state.
        (e.g. moving from counties to states)
    :param distrib: str, default="gaussian"
        the kind of distrib "poisson" or "gamma" or other (Gaussian),
        to compute the confidence
        intervals accordingly
    :param n_bins: int, default: 7
        number of bin for auto-binning (discretizing the values)
    :param geoid: str, default: 'INS'
        the name of the geoid columns, INS refers to the Belgian french name
    :param weight: None or str
        column (if any) of sample weights
    :param shp_file: geopandas frame
        the shapefile. E.G: load_be_shp()
    :param autobin: boolean, default=False
        whether or not to bin the data.

    :return: geopandas dataframe
    """
    # sanity checks

    if not isinstance(shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError(
            "The shapefile should be a " "geopandas.geodataframe.GeoDataFrame"
        )

    geom_merc = shp_file.copy()

    if cols_to_plot is not None:
        if not isinstance(cols_to_plot, list):
            raise TypeError("'cols_to_plot' should be a list of strings or None")

    if (predicted is None) and (cols_to_plot is not None):
        predicted = cols_to_plot

    # set style, ignore geopandas warnings
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    set_my_plt_style(height=3, width=3)

    # aggregate either by nis or by borough
    if dissolve_on is not None:
        if dissolve_on not in df.columns:
            raise KeyError(
                "the key `dissolve_on` you try to dissolve on is not in df.columns"
            )

        df_average, df_ = single_predictor_aggr_multi_mod(
            feature=dissolve_on,
            df=df,
            weight=weight,
            predicted=predicted,
            target=target,
            ci=False,
            autobin=autobin,
            distrib=distrib,
            n_vals=50,
            n_bins=n_bins,
        )

        geo_df = geom_merc.dissolve(by=dissolve_on).reset_index()
        geo_df = pd.merge(
            geo_df,
            df_.fillna(0),
            how="left",
            left_on=[dissolve_on],
            right_on=[dissolve_on],
        )
    else:
        df_average, df_ = single_predictor_aggr_multi_mod(
            feature="geoid",
            df=df,
            weight=weight,
            predicted=predicted,
            target=target,
            ci=False,
            autobin=autobin,
            distrib=distrib,
            n_vals=50,
            n_bins=n_bins,
        )

        geo_df = geom_merc.reset_index()
        df_ = df_.fillna(0)
        df_["geoid"] = df_["geoid"].astype(str)
        geo_df[geoid] = geo_df[geoid].astype(str)
        geo_df = pd.merge(
            geo_df, df_.fillna(0), how="left", left_on=[geoid], right_on=["geoid"]
        )

    geo_df = gpd.GeoDataFrame(geo_df)

    return gpd.GeoDataFrame(geo_df)


def plot_on_map(
    df,
    target,
    dissolve_on=None,
    distrib="gaussian",
    plot_uncertainty=False,
    plot_weight=False,
    autobin=False,
    n_bins=7,
    geoid="nis",
    weight=None,
    shp_file=None,
    figsize=(12, 12),
    cmap=None,
    normalize=True,
    facecolor="black",
    nbr_of_dec=None,
):
    """
    Prepare the geodata to map. Take the dataframe df,
    which should have a geoid column, and join it to
    the zipcode mapper and the geometries.
    The result is illustrated on a (facet) chart.

    An aggregation step by geoid is performed in order to have a
    value per unique geoid.

    If weight is provided, then weighted average, at the geoid level,
    of the target is computed. The uncertainty
    (on the weighted average) is computed accordingly to the
    distribution (gaussian, Poisson or Gamma) and can be illustrated
    in another panel and the sample weights as well.

    If dissolve it set to True, the geometries are "dissolved" to go
    at the upper level, e.g. dissolve counties to go at the state level
    (a state being larger and containing several counties).

    :param df: dataframe
        the data set, predictors + dependent variable
    :param target: str
        the target name (main column to plot, e.g. observed values/truth)
    :param dissolve_on: str
        the column name you want to dissolve on. Dissolve
         means going to an upper geographical level,
        e.g. from commune to district or to state.
        (e.g. moving from counties to states)
    :param distrib: str, default='gaussian'
        the distribution for computing the confidence intervals. E
        ither 'gaussian', 'poisson' or 'gamma'.
    :param plot_uncertainty: Boolean, default=True
        whether or not plot the uncertainty on the weighted average on the geoid level.
    :param plot_weight: Boolean, default=True
        whether or not plot the weight on the geoid level.
    :param autobin: Bool, default=False
        autobin (discretized) the illustrated values. If True,
        the values are binned using percentiles before to be
        plotted
    :param n_bins: int, default: 7
        number of bin for auto-binning (discretizing the values)
    :param geoid: str, default: 'INS'
        the name of the geoid columns, INS refers to the Belgian french name
    :param weight: None or str
        column (if any) of sample weights
    :param shp_file: geopandas frame
        the shapefile. E.G: load_be_shp
    :param figsize: 2-uple, default: (12, 12)
        figure size, (width, heihgt)
    :param cmap: matplotlib cmap or None
        Please, use ONLY scientific color maps. No jet, no rainbow!
        If you have any doubt, keep the default
        e.g. http://www.fabiocrameri.ch/colourmaps.php
    :param normalize: bool
        should the other illustrated columns normalized to the target?
        If True, the vmin and vmax will be the same for all the panels
        and computed w.r.t the target column
    :param facecolor: str or 3-uple rgb
        the facecolor
    :param nbr_of_dec: int, default = None
        the number of decimal in the discrete legend, if autobin is used.
        If None, either 2 or 4 decimals

    :return: object, matplotlib figure
    """

    if not isinstance(shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError("The shapefile should be a geopandas.geodataframe.GeoDataFrame")

    title_col = "white" if facecolor == "black" else "black"
    # load the data to illustrate
    geo_df = prepare_geo_data(
        df=df,
        cols_to_plot=None,
        target=target,
        predicted=None,
        dissolve_on=dissolve_on,
        n_bins=n_bins,
        geoid=geoid,
        weight=weight,
        shp_file=shp_file,
        distrib=distrib,
    )

    geo_df["2target_std"] = 2 * geo_df["target_std"]

    # if more than 1 column to illustrate, fillna and
    # set the number of rows in the panel plot
    if plot_uncertainty and plot_weight:
        ncols = 3
    elif (plot_uncertainty and plot_weight is False) or (
        plot_uncertainty is False and plot_weight
    ):
        ncols = 2
    else:
        ncols = 1

    # Create figure and axes (this time it's 9, arranged 3 by 3)
    f, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, facecolor=facecolor)

    # delete non-used axes
    n_charts = ncols
    n_subplots = 1 * ncols
    # if n_charts-1 < (n_rows * ncols):
    #     f.delaxes(axs[n_rows - 1:, n_charts - 1])

    # Make the axes accessible with single indexing
    if n_charts > 1:
        axs = axs.flatten()

    # define colour_map
    if (cmap is None) and (facecolor == "black"):
        colour_map = cmr.tropical
    elif (cmap is None) and (facecolor != "black"):
        colour_map = cmr.bubblegum
    else:
        colour_map = cmap

    # loop over the columns to illustrate
    cols_to_enum = [target]

    if plot_uncertainty:
        cols_to_enum += ["2target_std"]

    if plot_weight and (weight is not None):
        cols_to_enum += [weight]
    elif plot_weight and (weight is None):
        cols_to_enum += ["weight"]

    for i, col in enumerate(cols_to_enum):
        # select the axis where the map will go
        if n_charts > 1:
            ax = axs[i]
        else:
            ax = axs

        # define the bins
        if autobin:
            # normalize or not to the reference column (here: target)
            if normalize and col != weight:
                bins = Quantiles(geo_df[target].fillna(0), n_bins).bins
            else:
                bins = Quantiles(geo_df[col].fillna(0), n_bins).bins

            if nbr_of_dec is not None:
                nbr_dec = str(nbr_of_dec)
            elif (np.abs(bins) < 1).all():
                nbr_dec = str(4)
            else:
                nbr_dec = str(2)

            geo_df.plot(
                column=col,
                ax=ax,
                legend=True,
                linewidth=0,
                cmap=colour_map,
                scheme="user_defined",
                classification_kwds={"bins": bins},
                legend_kwds={
                    "facecolor": facecolor,
                    "framealpha": 0,
                    "loc": "lower left",
                    "fmt": "{:." + nbr_dec + "f}",
                    "labelcolor": title_col,
                },
            )
        else:
            if normalize:
                vmax = np.nanpercentile(geo_df[target].fillna(0), 99)
                vmin = np.nanpercentile(geo_df[target].fillna(0), 1)
            else:
                vmax = np.nanpercentile(geo_df[col].fillna(0), 99)
                vmin = np.nanpercentile(geo_df[col].fillna(0), 1)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cax.tick_params(
                axis="y", labelsize="large", labelcolor=title_col, grid_linewidth=0
            )
            cax.set_frame_on(False)
            geo_df.plot(
                column=col,
                ax=ax,
                cax=cax,
                legend=True,
                linewidth=0,
                vmin=vmin,
                vmax=vmax,
                cmap=colour_map,
            )
        # Remove axis clutter
        ax.set_axis_off()
        # Set the axis title to the name of variable being plotted

        if col == "2target_std":
            ax.set_title("2 standard dev.", color=title_col)
        else:
            ax.set_title(col, color=title_col)

    if n_subplots > n_charts > 1:
        for i in range(n_charts, n_subplots):
            ax = axs[i]
            ax.set_axis_off()

    # Display the figure
    # plt.show()
    return f


def facet_map(
    df,
    cols_to_plot,
    target,
    predicted=None,
    dissolve_on=None,
    autobin=False,
    n_bins=7,
    geoid="nis",
    weight=None,
    shp_file=None,
    figsize=(12, 12),
    ncols=2,
    cmap=None,
    normalize=True,
    facecolor="black",
    nbr_of_dec=None,
):
    """
    Prepare the geodata to map. Take the dataframe df,
    which should have a geoid column, and join it to
    the zipcode mapper and the geometries.
    The result is illustrated on a (facet) chart.

    An aggregation step by geoid is performed in
    order to have a value per unique geoid.

    If weight is provided, then weighted average,
    at the geoid level, of the target is computed.

    If dissolve it set to True, the geometries are "dissolved"
    to go at the upper level, e.g. dissolve counties to
    go at the state level (a state being larger and containing
    several counties).

    :param df: pandas dataframe
        the data set, predictors + dependent variable
    :param cols_to_plot: list of str of None
        the columns to plot with the target (e.g. predicted values
        or any other columns than target)
    :param target: str
        the target name (main column to plot, e.g. observed values/truth)
    :param predicted: str or pandas data frame or None, default: None
        if dataframe, should be the same length of df.
        It could be predicted values, not joined to df
    :param dissolve_on: str
        the column name you want to dissolve on.
        Dissolve means going to an upper geographical level,
        e.g. from commune to district or to state.
        (e.g. moving from counties to states)
    :param autobin: Bool, default=False
        autobin (discretized) the illustrated values.
        If True, the values are binned using percentiles before to be plotted
    :param n_bins: int, default: 7
        number of bin for auto-binning (discretizing the values)
    :param geoid: str, default: 'INS'
        the name of the geoid columns, INS refers to the Belgian french name
    :param weight: None or str
        column (if any) of sample weights
    :param shp_file: geopandas frame
        the shapefile. E.G: load_be_shp()
    :param figsize: 2-uple, default: (12, 12)
        figure size, (width, heihgt)
    :param ncols: int, default: 2
        the number of columns in the facet plot, if there are several columns to plot
    :param cmap: matplotlib cmap or None
        Please, use ONLY scientific color maps. No jet, no rainbow!
        If you have any doubt, keep the default
        e.g. http://www.fabiocrameri.ch/colourmaps.php
    :param normalize: bool
        should the other illustrated columns normalized to the target?
        If True, the vmin and vmax will be
        the same for all the panels and computed w.r.t the target column
    :param facecolor: str or 3-uple rgb
        the facecolor
    :param nbr_of_dec: int, default = None
        the number of decimal in the discrete legend,
        if autobin is used. If None, either 2 or 4 decimals

    :return: object, matplotlib figure
    """

    if not isinstance(ncols, int):
        raise ValueError("'ncols' should be an integer")

    if not isinstance(shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError("The shapefile should be a geopandas.geodataframe.GeoDataFrame")

    title_col = "white" if facecolor == "black" else "black"

    if facecolor == "black":
        set_my_plt_style(height=3, width=5, linewidth=2, bckgnd_color=facecolor)

    # load the data to illustrate
    geo_df = prepare_geo_data(
        df=df,
        cols_to_plot=cols_to_plot,
        target=target,
        predicted=predicted,
        dissolve_on=dissolve_on,
        n_bins=n_bins,
        geoid=geoid,
        weight=weight,
        shp_file=shp_file,
    )

    # if more than 1 column to illustrate, fillna and set the
    # number of rows in the panel plot
    if cols_to_plot is not None:
        geo_df[cols_to_plot] = geo_df[cols_to_plot]#.fillna(0)
        n_rows = int(np.ceil((len(cols_to_plot) + 1) / ncols))
        ncols_to_plot = len(cols_to_plot + [target])
    else:
        n_rows = 1
        ncols_to_plot = 1
        ncols = 1

    # Create figure and axes (this time it's 9, arranged 3 by 3)
    f, axs = plt.subplots(
        nrows=n_rows, ncols=ncols, figsize=figsize, facecolor=facecolor
    )

    # delete non-used axes
    n_charts = ncols_to_plot
    n_subplots = n_rows * ncols
    # if n_charts-1 < (n_rows * ncols):
    #     f.delaxes(axs[n_rows - 1:, n_charts - 1])

    # Make the axes accessible with single indexing
    if n_charts > 1:
        axs = axs.flatten()

    # define colour_map
    if (cmap is None) and (facecolor == "black"):
        colour_map = cmr.tropical
    elif (cmap is None) and (facecolor != "black"):
        colour_map = cmr.bubblegum
    else:
        colour_map = cmap

    # loop over the columns to illustrate

    if cols_to_plot is not None:
        cols_to_enum = [target] + cols_to_plot
    else:
        cols_to_enum = [target]

    for i, col in enumerate(cols_to_enum):
        # select the axis where the map will go
        if n_charts > 1:
            ax = axs[i]
        else:
            ax = axs

        # define the bins
        if autobin:
            # normalize or not to the reference column (here: target)
            if normalize:
                bins = Quantiles(geo_df[target].fillna(0), n_bins).bins
            else:
                bins = Quantiles(geo_df[col].fillna(0), n_bins).bins

            if nbr_of_dec is not None:
                nbr_dec = str(nbr_of_dec)
            elif (np.abs(bins) < 1).all():
                nbr_dec = str(4)
            else:
                nbr_dec = str(2)

            geo_df.plot(
                column=col,
                ax=ax,
                legend=True,
                linewidth=0,
                cmap=colour_map,
                scheme="user_defined",
                classification_kwds={"bins": bins},
                legend_kwds={
                    "facecolor": facecolor,
                    "framealpha": 0,
                    "loc": "lower left",
                    "fmt": "{:." + nbr_dec + "f}",
                    "labelcolor": title_col,
                },
            )
        else:
            if normalize:
                vmax = np.nanpercentile(geo_df[target].fillna(0), 99)
                vmin = np.nanpercentile(geo_df[target].fillna(0), 1)
            else:
                vmax = np.nanpercentile(geo_df[col].fillna(0), 99)
                vmin = np.nanpercentile(geo_df[col].fillna(0), 1)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cax.tick_params(
                axis="y",
                labelsize="large",
                labelcolor=title_col,
                colors=title_col,
                grid_linewidth=0,
            )
            cax.set_frame_on(False)
            geo_df.plot(
                column=col,
                ax=ax,
                cax=cax,
                legend=True,
                linewidth=0,
                vmin=vmin,
                vmax=vmax,
                cmap=colour_map,
            )
        # Remove axis clutter
        ax.set_axis_off()
        # Set the axis title to the name of variable being plotted

        ax.set_title(col, color=title_col)

    if n_subplots > n_charts > 1:
        for i in range(n_charts, n_subplots):
            ax = axs[i]
            ax.set_axis_off()

    # Display the figure
    # plt.show()
    return f


def facet_map_interactive(
    df,
    cols_to_plot,
    target,
    predicted=None,
    dissolve_on=None,
    alpha=0.5,
    autobin=False,
    n_bins=7,
    geoid="nis",
    weight="exp_yr",
    shp_file=None,
    figsize=(12, 12),
    ncols=2,
    cmap=None,
    normalize=True,
    tiles_src=None,
):
    """
    Prepare the geodata to map. Take the dataframe df,
    which should have a geoid column, and join it to
    the zipcode mapper and the geometries.
    The result is illustrated on a (facet) chart.

    An aggregation step by geoid is performed in order to have a
    value per unique geoid.

    If weight is provided, then weighted average, at the geoid level,
    of the target is computed. The uncertainty (on the weighted average)
    is computed accordingly to the distribution (gaussian, Poisson or Gamma)
    and can be illustrated in another panel and the sample weights as well.

    If dissolve it set to True, the geometries are "dissolved"
    to go at the upper level, e.g. dissolve counties to go at the state level
    (a state being larger and containing several counties).



    :param df: dataframe
        the data set, predictors + dependent variable
    :param cols_to_plot: list of str of None
        the columns to plot with the target (e.g. predicted values
        or any other columns than target)
    :param target: str
        the target name (main column to plot, e.g. observed values/truth)
    :param predicted: str or pandas data frame or None, default: None
        if dataframe, should be the same length of df.
        It could be predicted values, not joined to df
    :param dissolve_on: str
        the column name you want to dissolve on. Dissolve means going to
        an upper geographical level, e.g. from commune to district or to state.
        (e.g. moving from counties to states)
    :param alpha: float between 0 and 1, default=0.5
        the transparency of the choropleth
    :param autobin: Bool, default=False
        autobin (discretized) the illustrated values.
        If True, the values are binned using percentiles before to be plotted
    :param n_bins: int, default: 7
        number of bin for auto-binning (discretizing the values)
    :param geoid: str, default: 'INS'
        the name of the geoid columns, INS refers to the Belgian french name
    :param weight: None or str
        column (if any) of sample weights
    :param shp_file: geopandas frame
        the shapefile. E.G: load_be_shp()
    :param figsize: 2-uple, default: (12, 12)
        figure size, (width, heihgt)
    :param ncols: int, default: 2
        the number of columns in the facet plot, if there are several columns to plot
    :param cmap: matplotlib cmap or None
        Please, use ONLY scientific color maps. No jet, no rainbow!
        If you have any doubt, keep the default
        e.g. http://www.fabiocrameri.ch/colourmaps.php
    :param normalize: bool
        should the other illustrated columns normalized to the target?
        If True, the vmin and vmax will be the same for all the panels
        and computed w.r.t the target colum,
    :param tiles_src: str or None
        the tile source, should be a string, compatible with one of the geoview tile sources

    :return: matplotlib figure
    """

    if not isinstance(ncols, int):
        raise ValueError("'ncols' should be an integer")

    # load the data to illustrate
    geo_df = prepare_geo_data(
        df=df,
        cols_to_plot=cols_to_plot,
        target=target,
        predicted=predicted,
        dissolve_on=dissolve_on,
        n_bins=n_bins,
        geoid=geoid,
        weight=weight,
        shp_file=shp_file,
        autobin=False,
    )

    # if more than 1 column to illustrate, fillna and set
    # the number of rows in the panel plot
    if cols_to_plot is not None:
        geo_df[cols_to_plot] = geo_df[cols_to_plot].fillna(0)
    else:
        cols_to_plot = []

    if (tiles_src is not None) and (
        tiles_src not in list(hv.element.tiles.tile_sources.keys())
    ):
        raise ValueError(
            "`tiles_src` should be a string and one "
            "of {}".format(list(hv.element.tiles.tile_sources.keys()))
        )
    elif (tiles_src is not None) and (
        tiles_src in list(hv.element.tiles.tile_sources.keys())
    ):
        tiles = hv.element.tiles.tile_sources[tiles_src]()
    else:
        tiles = hv.element.tiles.tile_sources["CartoLight"]()

    # define colour_map
    if (cmap is None) and (autobin is False):
        colour_map = cmr.tropical  # Thermal_20.mpl_colormap
    elif (cmap is None) and (autobin is True):
        colour_map = cmr.tropical
    else:
        colour_map = cmap

    hv_plot_list = []
    # loop over the columns to illustrate
    for i, col in enumerate([target] + cols_to_plot):

        plot_opts = dict(
            tools=["hover"],
            width=550,
            height=450,
            color_index=col,
            cmap=colour_map,
            colorbar=True,
            toolbar="above",
            xaxis=None,
            yaxis=None,
            alpha=alpha,
            title=col,
            clipping_colors={"NaN": "white"},
        )

        if autobin:
            if normalize:
                ser, bins = pd.qcut(
                    geo_df[target].fillna(0), q=n_bins, retbins=True, labels=None, precision=2
                )
                geo_df[col] = (
                    pd.cut(
                        geo_df[col].fillna(0),
                        bins=bins,
                        labels=np.unique(ser),
                        include_lowest=True,
                    )
                    .apply(lambda x: x.mid)
                    .astype(float)
                )
                # pd.cut(geo_df[col], q=n_bins, duplicates='drop').
                # apply(lambda x: x.mid).astype(float)
            else:
                geo_df[col] = (
                    pd.qcut(geo_df[col], q=n_bins, duplicates="drop", precision=2)
                    .apply(lambda x: x.mid)
                    .astype(float)
                )
                # .apply(lambda x: x.mid).astype(float)

            hv_plot_list.append(
                tiles
                * gv.Polygons(
                    geo_df, vdims=[hv.Dimension(col)], crs=ccrs.GOOGLE_MERCATOR
                ).opts(**plot_opts)
            )
        else:
            if normalize:
                vmax = np.nanpercentile(geo_df[target].fillna(0), 99)
                vmin = np.nanpercentile(geo_df[target].fillna(0), 1)
            else:
                vmax = np.nanpercentile(geo_df[col].fillna(0), 99)
                vmin = np.nanpercentile(geo_df[col].fillna(0), 1)

            hv_plot_list.append(
                tiles
                * gv.Polygons(
                    geo_df,
                    vdims=[hv.Dimension(col, range=(vmin, vmax))],
                    crs=ccrs.GOOGLE_MERCATOR,
                ).opts(**plot_opts)
            )

    # Display the figure
    hvl = (
        hv.Layout(hv_plot_list)
        .opts(opts.Tiles(width=figsize[0], height=figsize[1]))
        .cols(2)
    )
    return hvl


def load_geometry(shp_path, geoid="INS"):
    """
    Load the shapefiles and set the coordinates and projection
    :param geoid: str
        the name of the geoid column
    :param shp_path: str
        path to the shape file
    :return:
     geom_merc: geopandas dataframe
        the geopandas df to plot the map
    """
    geometries = gpd.read_file(shp_path)  # self.path_dic['shp_path'])
    geom_merc = geometries.to_crs(ccrs.GOOGLE_MERCATOR.proj4_init)
    geom_merc[geoid] = geom_merc[geoid].astype(str)
    return geom_merc


def load_be_shp():
    """
    Load the shapefile of the Greatest Country in the Universe.

    :return: geopandas dataframe
    """
    # module_path = dirname(__file__)
    # base_dir = join(module_path, "beshp")
    # data_filename = join(base_dir, "belgium.shp")
    data_filename = resource_filename(__name__, 'beshp/belgium.shp')
    return gpd.read_file(data_filename)


def set_my_plt_style(height=3, width=5, linewidth=2, bckgnd_color="#f5f5f5"):
    """
    This set the style of matplotlib to fivethirtyeight
    with some modifications (colours, axes)

    :param linewidth: float, default=2
        line width
    :param height: float, default=3
        fig height
    :param width: float, default=5
        fig width
    :param bckgnd_color: str, default="#f5f5f5"
        the background color

    :return: Nothing
    """
    plt.style.use("fivethirtyeight")
    my_colors_list = Bold_10.hex_colors
    myorder = [2, 3, 4, 1, 0, 6, 5, 8, 9, 7]
    my_colors_list = [my_colors_list[i] for i in myorder]
    params = {
        "figure.figsize": (width, height),
        "axes.prop_cycle": plt.cycler(color=my_colors_list),
        "axes.facecolor": bckgnd_color,
        "patch.edgecolor": bckgnd_color,
        "figure.facecolor": bckgnd_color,
        "axes.edgecolor": bckgnd_color,
        "savefig.edgecolor": bckgnd_color,
        "savefig.facecolor": bckgnd_color,
        "grid.color": "#d2d2d2",
        "lines.linewidth": linewidth,
    }  # plt.cycler(color=my_colors_list)
    mpl.rcParams.update(params)
