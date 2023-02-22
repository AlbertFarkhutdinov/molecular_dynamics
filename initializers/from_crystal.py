from typing import Optional

import numpy as np

from common.helpers import get_empty_vectors
from configurations import ThermodynamicSystem
from initializers.base_initializer import BaseInitializer


class FromCrystal(BaseInitializer):

    def __init__(
            self,
            crystal_type: str,
            lattice_constant: float,
            particles_numbers: tuple[int, int, int],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.crystal_type = crystal_type
        self.lattice_constant = lattice_constant
        self.particles_numbers = np.array(particles_numbers)

    def _get_thermodynamic_system(self):
        lattice_points = self.__get_lattice_points()
        cell_dimensions = (
                self.particles_numbers
                * lattice_points
                * self.lattice_constant
        )
        system = ThermodynamicSystem(
            positions=self.__generate_ordered_state(lattice_points),
            cell_dimensions=cell_dimensions,
            is_pbc_applied=True,
        )
        return system

    def __get_lattice_points(self) -> Optional[int]:
        _lattice_points_numbers = {
            'PC': 1,
            'BCC': 2,
            'FCC': 4,
        }
        if self.crystal_type in _lattice_points_numbers:
            return _lattice_points_numbers[self.crystal_type]
        raise KeyError('Unacceptable `crystal_type`.')

    def __generate_ordered_state(self, lattice_points: int) -> np.array:
        positions = get_empty_vectors(self.particles_numbers.prod())
        r_cell = get_empty_vectors(lattice_points)
        if self.crystal_type:
            r_cell[1][0] = 0.5
            r_cell[3][0] = 0.5
            r_cell[1][1] = 0.5
            r_cell[2][1] = 0.5
            r_cell[2][2] = 0.5
            r_cell[3][2] = 0.5
        position_number = 0
        for k in range(self.particles_numbers[2]):
            for j in range(self.particles_numbers[1]):
                for i in range(self.particles_numbers[0]):
                    for point in range(lattice_points):
                        position_number += 1
                        positions[position_number - 1] = (
                                self.lattice_constant *
                                (np.array([i, j, k]) + r_cell[point])
                        )
        self.logger.debug('System center is set to zero.')
        positions -= positions.sum(axis=0) / self.particles_numbers.prod()
        return positions
