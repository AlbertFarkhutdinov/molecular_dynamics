from datetime import datetime
from typing import Dict, Iterable

import numpy as np


def get_empty_vectors(vectors_number: int) -> np.ndarray:
    return np.zeros((vectors_number, 3), dtype=np.float)


def get_empty_float_scalars(scalars_number: int) -> np.ndarray:
    return np.zeros(scalars_number, dtype=np.float)


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
