from typing import Iterable

import numpy as np


class SystemStaticParameters:

    def __init__(self, **static_parameters):
        self.init_type = static_parameters.get('init_type')
        if self.init_type == -1:
            self.particles_number = 0
            self.cell_dimensions = np.zeros(3, dtype=np.float)
        elif self.init_type in (1, 2):
            self.lattice_constant = static_parameters.get('lattice_constant')
            nb_dict = {
                'пк': 1,
                'гцк': 4,
            }
            if static_parameters.get('crystal_type') in nb_dict:
                self.crystal_type = static_parameters.get('crystal_type')
                self.nb = nb_dict[self.crystal_type]
            else:
                raise KeyError('Unacceptable `crystal_type`.')

            particles_number = static_parameters.get('particles_number')
            if self.init_type == 1 and isinstance(particles_number, Iterable):
                self.particles_numbers = np.array(particles_number, dtype=np.int)
                self.cell_dimensions = np.array(particles_number, dtype=np.float) * self.lattice_constant
                self.particles_number = self.nb * self.particles_numbers.prod()
            elif self.init_type == 2 and isinstance(particles_number, int):
                self.particles_number = particles_number
                self.cell_dimensions = (
                        np.ones(3, dtype=np.float)
                        * self.lattice_constant
                        * np.round((particles_number / self.nb) ** (1/3))
                )
            else:
                raise TypeError('Conflict between `init_type` and `particles_number`.')
        else:
            raise ValueError('Unacceptable `init_type`.')

    def get_cell_volume(self) -> float:
        return self.cell_dimensions.prod()

    def get_density(self, volume=None) -> float:
        _volume = volume or self.get_cell_volume()
        return self.particles_number / _volume
