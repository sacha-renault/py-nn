from typing import Any

def wrap_as_list(data: Any | list[Any], type_: type) -> list:
    # wrap as list
    if not isinstance(data, (list, tuple)):
        data = [ data ]

    # ensure all data are defined type
    if not all([isinstance(x, type_) for x in data]):
        raise TypeError(f"All instances of list must be type : {type_.__name__}")

    return data
