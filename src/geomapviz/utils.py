# Settings and libraries
from __future__ import print_function
import pandas as pd
from typing import List


def check_list_of_str(str_list: List[str], name: str = "str_list") -> None:
    """Raise an exception if ``str_list`` is not a list of strings

    Parameters
    ----------
    str_list :

    name :
        (default ``'str_list'``)

    Raises
    ------
    TypeError
        if ``str_list`` is not a ``List[str]``

    """
    if str_list is not None:
        if not (
            isinstance(str_list, list) and all(isinstance(s, str) for s in str_list)
        ):
            raise TypeError(f"{name} must be a list of one or more strings.")


def convert_category_to_code(df: pd.DataFrame):
    """convert_category_to_code converts categories (levels) to codes for easier representation

    Parameters
    ----------
    df :
        dataframe with cateorical columns

    Returns
    -------
    pd.DataFrame
        numerical dataframe
    """
    cat_cols = df.select_dtypes("category").columns.tolist()
    if cat_cols:
        for c in cat_cols:
            df[c] = df[c].cat.codes
    return df
