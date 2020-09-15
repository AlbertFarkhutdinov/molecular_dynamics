import numpy as np


def get_empty_vectors(vectors_number: int) -> np.ndarray:
    return np.zeros((vectors_number, 3), dtype=np.float)


def get_empty_float_scalars(scalars_number: int) -> np.ndarray:
    return np.zeros(scalars_number, dtype=np.float)


def get_empty_int_scalars(scalars_number: int) -> np.ndarray:
    return np.zeros(scalars_number, dtype=np.int32)
