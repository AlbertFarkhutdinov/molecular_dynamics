from datetime import datetime
from json import load, dumps
from os.path import join
from typing import Any, Dict, Iterable, Optional, Set

import numpy as np

from constants import PATH_TO_CONFIG, PATH_TO_DATA


def get_config_parameters(file_name):
    _config_filename = join(
        PATH_TO_CONFIG,
        file_name
    )
    with open(_config_filename, encoding='utf8') as file:
        config_parameters = load(file)
    return config_parameters


def save_config_parameters(config_parameters, config_number):
    _config_filename = join(
        PATH_TO_DATA,
        f'{get_date()}_config_{config_number}.inf',
    )
    with open(_config_filename, encoding='utf8', mode='w') as file:
        file.write(dumps(config_parameters, indent=2, ensure_ascii=False))


def get_empty_vectors(vectors_number: int) -> np.ndarray:
    return np.zeros((vectors_number, 3), dtype=np.float32)


def get_empty_float_scalars(scalars_number: int) -> np.ndarray:
    return np.zeros(scalars_number, dtype=np.float32)


def get_empty_int_scalars(scalars_number: int) -> np.ndarray:
    return np.zeros(scalars_number, dtype=np.int32)


def sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def get_formatted_time():
    return datetime.now().strftime("%Y-%m-%d_%H_%M")


def get_date():
    return datetime.today().strftime("%Y-%m-%d")


def get_parameters_dict(names: Iterable, value_size: int):
    return {
        name: get_empty_float_scalars(value_size)
        for name in names
    }


def print_info(
        step: int,
        iterations_numbers: int,
        current_time: float,
        parameters: Dict[str, Iterable],
):
    print(
        f'Step: {step}/{iterations_numbers};',
        f'\tTime = {current_time:.3f};',
        f'\tT = {parameters["temperature"][step - 1]:.5f};',
        f'\tP = {parameters["pressure"][step - 1]:.5f};\n',
        sep='\n',
    )


def math_round(
        number: float,
        number_of_digits_after_separator: int = 0,
) -> float:
    """
    Return rounded float number.

    Parameters
    ----------
    number : float
        Number to be rounded.
    number_of_digits_after_separator
        Number of digits after
        decimal separator in `number`.

    Returns
    -------
    float
        Rounded float number.

    """
    _multiplier = int('1' + '0' * number_of_digits_after_separator)
    _number_without_separator = number * _multiplier
    _integer_part = int(_number_without_separator)
    _first_discarded_digit = int(
        (_number_without_separator - _integer_part) * 10
    )
    if _first_discarded_digit >= 5:
        _integer_part += 1
    result = _integer_part / _multiplier
    return result


def get_representation(
        instance: Any,
        excluded: Optional[Set[str]] = None,
        is_base_included: bool = False,
) -> str:
    """
    Return the 'official' string representation of `instance`.

    Parameters
    ----------
    instance : Any
        The instance, which representation is returned.
    excluded : set, optional
        Names of arguments that are excluded
        from the representation.
    is_base_included : bool, optional, default: False
        If it is True, arguments of base class are included.

    Returns
    -------
    str
        The 'official' string representation of `instance`.

    """
    _parent_class_name = instance.__class__.__bases__[0].__name__
    _class_name = instance.__class__.__name__
    representation = [
        f'{_class_name}(',
    ]
    for _key, _value in instance.__dict__.items():
        _key_repr, _value_repr = _key, _value
        if _key.startswith(f'_{_parent_class_name}__'):
            if is_base_included:
                _key_repr = _key[3 + len(_parent_class_name):]
            else:
                continue
        if _key.startswith(f'_{_class_name}__'):
            _key_repr = _key[3 + len(_class_name):]
        if excluded and _key_repr in excluded:
            continue
        if isinstance(_value, (tuple, list, np.ndarray)):
            _value_repr = tuple(_value)
        representation.append(f'{_key_repr}={_value_repr!r}')
        representation.append(', ')
    if len(representation) > 1:
        representation.pop()
    representation.append(')')

    return ''.join(representation)
