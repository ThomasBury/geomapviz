"""Shared metric-list validation."""


def check_list_of_str(str_list: list[str], name: str = "str_list") -> None:
    if str_list is not None and not (
        isinstance(str_list, list) and all(isinstance(s, str) for s in str_list)
    ):
        raise TypeError(f"{name} must be a list of one or more strings.")
