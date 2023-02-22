from datetime import datetime
from json import load, dumps
from os.path import join
from typing import Iterable

import numpy as np
import pandas as pd

from common.constants import CONFIG_DIR, DATA_DIR


def get_json(*args):
    with open(join(str(CONFIG_DIR), *args), encoding='utf8') as file:
        config_parameters = load(file)
    return config_parameters


def save_config_parameters(config_parameters, config_number):
    _config_filename = str(
        DATA_DIR
        / get_date()
        / f'{get_date()}_config_{config_number}.inf',
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


def get_current_time():
    return datetime.now()


def get_formatted_time():
    return get_current_time().strftime("%Y-%m-%d_%H_%M")


def get_date():
    return get_current_time().strftime("%Y-%m-%d")


def get_parameters_dict(names: Iterable, value_size: int):
    return {
        name: get_empty_float_scalars(value_size)
        for name in names
    }


def print_info(
        step: int,
        iterations_numbers: int,
        current_time: float,
        parameters: dict[str, Iterable],
        step_index: int = None,
):
    _step_index = step_index or step
    print(
        f'Step: {step}/{iterations_numbers};',
        f'\tTime = {current_time:.3f};',
        f'\tT = {parameters["temperature"][step_index - 1]:.5f};',
        f'\tP = {parameters["pressure"][step_index - 1]:.5f};\n',
        sep='\n',
    )


def get_unique_frame(array: np.ndarray) -> pd.DataFrame:
    unique, counts = np.unique(array, axis=0, return_counts=True)
    return (
        pd.DataFrame(unique, counts)
        .reset_index()
        .rename(columns={'index': 'counts'})
        .sort_values(by='counts', ascending=False)
        .reset_index(drop=True)
    )
