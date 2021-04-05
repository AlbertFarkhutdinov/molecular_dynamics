from abc import ABC, abstractmethod


class BasePotential(ABC):

    def __init__(self):
        self.sigma = 1.0
        self.epsilon = 1.0

    @abstractmethod
    def get_energies_and_forces(self):
        # forces - это величины -dU/rdr.
        # Сила получается умножением на вектор r_ij
        raise NotImplementedError(
            'Define `get_energies_and_forces` in'
            f'{self.__class__.__name__}.'
        )
