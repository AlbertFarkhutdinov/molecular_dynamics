from typing import Iterable, Union

import numpy as np


class SystemStaticParameters:

    def __init__(
            self,
            init_type: int,
            lattice_constant: float,
            particles_number: Union[int, Iterable],
            crystal_type='пк',
    ):
        self.init_type = init_type
        self.lattice_constant = lattice_constant
        self.crystal_type = crystal_type
        self.nb = 4 if self.crystal_type == 'гцк' else 1
        if init_type == 1 and isinstance(particles_number, Iterable):
            self.particles_numbers = np.array(particles_number, dtype=np.int)
            self.cell_dimensions = np.array(particles_number, dtype=np.float) * self.lattice_constant
            self.particles_number = self.nb * self.particles_numbers.prod()
        elif init_type == 2 and isinstance(particles_number, int):
            self.particles_number = 1372
            self.cell_dimensions = (
                    np.ones(3, dtype=np.float)
                    * np.round((1372 / self.nb) ** (1/3)).astype(np.int)
                    * self.lattice_constant
            )
        else:
            raise TypeError('Conflict between `init_type` and `particles_number`.')

    def get_cell_volume(self) -> float:
        return self.cell_dimensions.prod()

    def get_density(self) -> float:
        return self.particles_number / self.get_cell_volume()
