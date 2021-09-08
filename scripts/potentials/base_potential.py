from abc import ABC, abstractmethod

import numpy as np


class BasePotential(ABC):

    def __init__(self, table_size: int, r_cut: float):
        self.sigma = 1.0
        self.epsilon = 1.0
        self.r_cut = r_cut
        self.table_size = table_size
        self.distances = 0.5 + 0.0001 * np.arange(1, self.table_size + 1)
        self.potential_table = self.get_energies_and_forces()

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @abstractmethod
    def get_energies_and_forces(self):
        # forces - это величины -dU/rdr.
        # Сила получается умножением на вектор r_ij
        raise NotImplementedError(
            'Define `get_energies_and_forces` in'
            f'{self.__class__.__name__}.'
        )
