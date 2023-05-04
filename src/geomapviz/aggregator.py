import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Optional, Union, Tuple, List, Dict, Literal
from pandas.api.types import is_numeric_dtype


from .utils import convert_category_to_code
from .utils import check_list_of_str


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns in the input DataFrame using the `.cat.codes` method.

    Parameters
    ----------
    df :
        Input DataFrame to encode categorical columns.

    Returns
    -------
    pd.DataFrame
        Returns a new DataFrame with categorical columns encoded.

    Examples
    --------
    >>> import pandas as pd
    >>> from typing import List
    >>> df = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c']),
                           'B': pd.Categorical(['b', 'a', 'b', 'c'], categories=['a', 'b', 'c']),
                           'C': [1, 2, 3, 4],
                           'D': [5, 6, 7, 8]})
    >>> encoded_df = encode_categorical_columns(df)
    >>> print(encoded_df)
       A  B  C  D
    0  0  1  1  5
    1  1  0  2  6
    2  2  1  3  7
    3  0  2  4  8
    """
    cat_cols = df.select_dtypes("category").columns.tolist()
    if cat_cols:
        for c in cat_cols:
            df[c] = df[c].cat.codes
    return df


def prepare_dataframe(df: pd.DataFrame, 
                      groups: Union[List[str], str], 
                      target: str,
                      other_cols_avg: Optional[List[str]] = None,
                      weight: Optional[str] = None, 
                      verb: int = 0, 
                      distr: str = "gaussian") -> pd.DataFrame:
    """
    Prepare dataframe for the confidence interval computation.

    Parameters
    ----------
    df : 
        Input data.
    groups : 
        List of column names containing the groups of interest.
    target : 
        Name of the target column.
    other_cols_avg :
        Other columns to average, such as the predicted values of a model
    weight : 
        Name of the weight column. Default is None.
    verb : 
        Controls the verbosity of the warning message. Default is 0.
    distr : 
        Name of the distribution. Default is "gaussian".

    Returns
    -------
    pd.DataFrame
        Prepared dataframe.

    Notes
    -----
    If weight is None, a weight column is added and set to 1.
    If the distribution is not Gaussian and the weight is not provided, a warning message is raised.
    """
    if isinstance(groups, str):
        groups = [groups]
        
    check_list_of_str(groups) 
    check_list_of_str(other_cols_avg)
    
    if other_cols_avg is None:
        other_cols_avg = []
    
    if weight is None:
        weight = "weight"
        df_ = df[groups + other_cols_avg + [target]].copy()
        df_[weight] = 1
        if verb > 0 and distr != "gaussian":
            warnings.warn(
                "Weight not provided, using the Gaussian approx for "
                "the CI. For the Poisson or Gamma ci, please provide weights (exposure or ncl)"
            )
    else:
        df_ = df[groups + other_cols_avg + [weight, target]].copy()
        
    # to count rows for each level
    df_["count"] = 1
    
    df_ = encode_categorical_columns(df_) 
    
    # for convenience and less complexity through the different function
    # let's rename the columns
    df_ = df_.rename(columns={target: "target", weight: "weight"}) 

    return df_

def compute_weighted_average(df: pd.DataFrame, 
                             groups: Union[str, List[str]],
                             target: str = "target", 
                             weight: str = "weight",
                             other_cols_avg: Optional[List[str]] = None
                             ):
    """compute_weighted_average computes the weighted arithmetic average, grouped by the column `group`.
    The weighted average is :math: `\sum_{i} w_{i} x_{i} / \sum_{i} w_{i}`
    If the weight is None, it computes the arithmetic average without weights :math: `\sum_{i} x_{i} / N`

    Parameters
    ----------
    df :
        the data set
    groups :
        the predictor(s) to group by
    target :
        the name of the observed/target column
    weight :
        the name of the column weight
    other_cols_avg :
        Other columns to average, such as the predicted values of a model
    Returns
    -------
    pd.DataFrame
        the dataframe with the arithmetic average, by group
    """
    
    # the weighted avg is sum(x_i * w_i) / sum(w_i * w_j)
    # this is the numerator
    df[target] = df[target] * df[weight]
    
    # check if str --> make a list
    if isinstance(groups, str):
        groups = [groups]
        
    check_list_of_str(groups) 
    
    if other_cols_avg is None:
        df = (
            df.groupby(groups)[[weight, target, "count"]]
            .sum()
            .reset_index()
            .assign(target=lambda x: x[target] / x[weight])
            )
        return df
    else:
        df[other_cols_avg] = df[other_cols_avg].values * np.expand_dims(df[weight].values, axis=-1)
        keep_cols = other_cols_avg  + [target, weight, "count"]
        
        df = (
            df.groupby(groups)[keep_cols]
            .sum()
            .reset_index()
            .assign(target=lambda x: x[target] / x[weight])
            )
        df[other_cols_avg] = df[other_cols_avg].values / np.expand_dims(df[weight].values, axis=-1)
        

        return df
     
def compute_confidence_interval(df: pd.DataFrame,
                                groups: Union[str, List[str]],
                                target: str = "target",
                                weight: str = "weight",  
                                other_cols_avg: Optional[List[str]] = None,
                                distr: str = "gaussian", 
                                n_std: float = 2.0):
    # check if str --> make a list
    if isinstance(groups, str):
        groups = [groups]
        
    check_list_of_str(groups) 
    selected_cols = groups + [target, weight, "count"]
    
    # update the list of selected columns if predictions are included
    if other_cols_avg:
        selected_cols = list(set(selected_cols).union(set(other_cols_avg)))

    df_long = pd.melt(
        df[selected_cols].copy(),
        id_vars=groups + [weight, "count"],
        var_name="model",
        value_name="avg",
    )

    if distr == "poisson":
        df_long["target_std"] = np.sqrt(df_long["avg"] / df_long[weight])
    elif distr == "gamma":
        df_long["target_std"] = df_long["avg"] * np.sqrt(1 / df_long[weight])
    elif distr == "gaussian":
        df_long["target_std"] = df_long["avg"] * np.sqrt(1 / df_long["count"])
    else:
        warnings.warn('distr is not in ["poisson", "gamma", "gaussian"], using Gaussian approx. for the conf. int.')
        df_long["target_std"] = df_long["avg"] * np.sqrt(1 / df_long["count"])
        
    df_long["ci_low"] = df_long["avg"] - n_std * df_long["target_std"]
    df_long["ci_low"] = df_long["ci_low"].clip(lower=0)
    df_long["ci_up"] = df_long["avg"] + n_std * df_long["target_std"]
    upper_bound = df_long["ci_up"].quantile(0.999)
    df_long["ci_up"] = df_long["ci_up"].clip(upper=upper_bound)
    df_long = df_long.reset_index()[["model"] + groups + ["avg", "ci_low", "ci_up", weight, "count"]]
    return df_long  
    
def weighted_average_aggregator(
    df: pd.DataFrame,
    groups: Union[str, List[str]],
    target: str,
    other_cols_avg: Optional[List[str]] = None,
    distr: str = "gaussian",
    weight: str = None,
    verb: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the weighted average and the confidence interval of a target variable
    in a Pandas DataFrame, grouped by one or more categorical columns.

    Parameters
    ----------
    df : 
        The input DataFrame to compute the weighted average and confidence interval on.
    groups : 
        The name(s) of the column(s) in `df` that define the groups to aggregate.
        If `groups` is a string, it will be interpreted as a single group column name.
        If `groups` is a list of strings, it will be interpreted as multiple group column names.
    target : 
        The name of the column in `df` that contains the target variable to aggregate.
    other_cols_avg : 
        The predicted values of the target variable to use for computing the confidence interval
        or any other columns to average.
        If `other_cols_avg` is not None, it should be a list of column names.
    distr : 
        The distribution to use for computing the confidence interval.
        Supported distributions are 'gaussian' (default), 't' and 'bootstrap'.
    weight : 
        The name of the column in `df` that contains the weights to use for computing the weighted average.
        If `weight` is None (default), all rows are assumed to have equal weight.
    verb : 
        Verbosity level of the function (0: no message, 1: info, 2: debug).
        The default is 0.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        A tuple of two DataFrames:
        - The first DataFrame contains the weighted average and the number of observations per group.
        - The second DataFrame contains the confidence interval of the weighted average, computed at 95% confidence level.

    Raises
    ------
    ValueError
        If any of the input arguments is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from my_module import weighted_average_aggregator
    >>> data = pd.DataFrame({'color': ['red', 'green', 'red', 'green', 'green'],
    ...                      'size': ['small', 'large', 'medium', 'large', 'small'],
    ...                      'price': [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> groups = ['color', 'size']
    >>> target = 'price'
    >>> weights = 'weights'
    >>> data[weights] = [1, 2, 3, 4, 5]
    >>> result, conf = weighted_average_aggregator(df=data, groups=groups, target=target, weight=weights)
    """  
    
    # check if str --> make a list
    if isinstance(groups, str):
        groups = [groups]
        
    check_list_of_str(groups) 
    
    df_ = prepare_dataframe(df=df,
                            groups=groups,
                            target=target,
                            other_cols_avg=other_cols_avg,
                            weight=weight,
                            verb=verb,
                            distr=distr)
            
    df_ = encode_categorical_columns(df_)  
        
    # for convenience and less complexity through the different function
    # let's rename the columns
    df_ = df_.rename(columns={target: "target", weight: "weight"})  
    
    df_= compute_weighted_average(df=df_, groups=groups, other_cols_avg=other_cols_avg, weight="weight", target="target")
    df_long = compute_confidence_interval(df=df_, groups=groups, other_cols_avg=other_cols_avg, distr=distr, n_std=2.0, weight="weight", target="target")
    
    return df_, df_long


def merge_zip_df(zip_path: str, df: pd.DataFrame, geoid: str = "geoid", cols_to_keep: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Merge a DataFrame `df` with a mapping table for the zipcode and other relevant geographical information
    (district name, sub-districts, etc.). The key is the `geoid` column.
    The zip mapper might be such as:

    +----+---------+-------------+---------+---------+------------+-----------+-------------------+
    |    |   geoid | town        |     lat |    long |   postcode | district  | borough           |
    +====+=========+=============+=========+=========+============+===========+===================+
    |  0 |   21004 | BRUSSEL     | 50.8333 | 4.35    |       1000 | Brussels  | Brussel Hoofdstad |
    +----+---------+-------------+---------+---------+------------+-----------+-------------------+
    |  1 |   21015 | SCHAARBEEK  | 50.85   | 4.38333 |       1030 | Brussels  | Brussel Hoofdstad |

    Parameters
    ----------
    zip_path :
        The path to the zipcode mapper, a csv file with additional geo info and a geoid column
    df : 
        The DataFrame to merge with the zipcode mapper
    geoid : 
        The name of the `geoid` column in both the `df` and the zipcode mapper
    cols_to_keep :
        The list of columns to keep from the zipcode mapper. If None, keep all columns.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame with additional geo information

    Raises
    ------
    TypeError
        If `cols_to_keep` is not None and not a list of strings

    """
    if (not isinstance(cols_to_keep, list)) and (cols_to_keep is not None):
        raise TypeError("If `cols_to_keep` is not None, it should be a list of strings")

    # Load zipcode mapper and adding borough to the DataFrame
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


def dissolve_and_aggregate(
    df: pd.DataFrame,
    target: str,
    other_cols_avg: Optional[List[str]] = None,
    dissolve_on: Optional[List[str]] = None,
    distr: str = "gaussian",
    geoid: str = "INS",
    weight: Optional[List[str]] = None,
    shp_file: Union[gpd.geodataframe.GeoDataFrame, None] = None,
    ) -> gpd.GeoDataFrame:
    """
    Dissolves a GeoDataFrame based on a column, and aggregates data based on the
    dissolved polygons.

    Parameters
    ----------
    df : 
        Dataframe with the data to be aggregated.
    cols_to_plot : 
        List of columns to plot on map.
    target : 
        Column with the target variable.
    other_cols_avg : 
        Columns with the predicted values or any other columns to average.
    distr : 
        Distribution of the target variable, by default "gaussian".
    weight : 
        Column with the weights to be used, by default None.
    dissolve_on : 
        Column to dissolve the GeoDataFrame, by default None.
    geoid : 
        Column with the geoid, by default "geoid".
    shp_file : 
        The shapefile to use for the map, as a GeoDataFrame. The default is None.

    Returns
    -------
    geopandas.GeoDataFrame
        geodataframe with the dissolved polygons.
    """
    # sanity checks

    if not isinstance(shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError("The shapefile should be a GeoDataFrame")
        
    geom_merc = shp_file.copy()
    
    if other_cols_avg and not isinstance(other_cols_avg, list):
        raise TypeError("'other_cols_avg' should be a list of strings or None")

    if dissolve_on:
        if dissolve_on not in df.columns:
            raise KeyError(f"{dissolve_on} is not a column in df")
        groups = dissolve_on
    else:
        groups = geoid

    df_, df_long = weighted_average_aggregator(
        df=df, groups=groups, target=target, other_cols_avg=other_cols_avg, distr=distr, weight=weight
    )

    if dissolve_on:
        geo_df = geom_merc.dissolve(by=dissolve_on).reset_index()
        merge_key = dissolve_on
    else:
        geo_df = geom_merc.reset_index()
        merge_key = geoid

    df_long = df_long.fillna(0)
    df_[merge_key] = df_[merge_key].astype(str)
    geo_df[merge_key] = geo_df[merge_key].astype(str)
    
    geo_df = geo_df.merge(df_long, left_on=merge_key, right_on=merge_key, how="left")

    return gpd.GeoDataFrame(geo_df)