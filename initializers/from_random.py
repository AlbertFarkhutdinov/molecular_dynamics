import numpy as np
from numpy.random import random

from common.helpers import get_empty_vectors
from common.numba_procedures import check_distances
from configurations import ThermodynamicSystem
from initializers.base_initializer import BaseInitializer


class FromRandom(BaseInitializer):

    def __init__(
            self,
            particles_number: int,
            cell_dimensions: tuple[float, float, float],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.particles_number = particles_number
        self.cell_dimensions = cell_dimensions

    def _get_thermodynamic_system(self):
        positions = self.__generate_random_state()
        system = ThermodynamicSystem(
            positions=positions,
            cell_dimensions=self.cell_dimensions,
            is_pbc_applied=True,
        )
        return system

    def __generate_random_state(self) -> np.array:
        np.random.seed(0)
        positions = get_empty_vectors(self.particles_number)
        positions[0] = random(3) * self.cell_dimensions
        for j in range(1, self.particles_number):
            is_distance_too_small = True
            while is_distance_too_small:
                positions[j] = random(3) * self.cell_dimensions
                is_distance_too_small = check_distances(
                    particle_index=j,
                    positions=positions,
                    cell_dimensions=self.cell_dimensions,
                )
        positions -= positions.sum(axis=0) / self.particles_number
        return positions
