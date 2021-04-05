from typing import Iterable, Optional, Union

import numpy as np

from scripts.potentials.lennard_jones import LennardJones
from scripts.potentials.dzugutov import Dzugutov


class ImmutableParameters:

    def __init__(self, **immutable_parameters):
        self.time_step = immutable_parameters.get('time_step')
        self.init_type = immutable_parameters.get('init_type')
        if self.init_type == -1:
            self.particles_number = 0
        elif self.init_type in (1, 2):
            self.nb = self.get_nb(
                crystal_type=immutable_parameters.get('crystal_type')
            )
            self.particles_numbers = None
            self.particles_number = self.get_particles_number(
                particles_number=immutable_parameters.get('particles_number')
            )
        else:
            raise ValueError('Unacceptable `init_type`.')
        self.potential_table = self.get_potential_table(
            potential_type=immutable_parameters.get('potential_type'),
        )
        self.skin_for_neighbors = immutable_parameters.get(
            'skin_for_neighbors'
        )

    @staticmethod
    def get_nb(crystal_type: str) -> Optional[int]:
        nb_dict = {
            'пк': 1,
            'гцк': 4,
        }
        if crystal_type in nb_dict:
            return nb_dict[crystal_type]
        raise KeyError('Unacceptable `crystal_type`.')

    def get_particles_number(
            self,
            particles_number: Union[int, Iterable],
    ) -> Optional[int]:
        if self.init_type == 1 and isinstance(particles_number, Iterable):
            self.particles_numbers = particles_number
            return self.nb * np.array(
                particles_number,
                dtype=np.int,
            ).prod()
        elif self.init_type == 2 and isinstance(particles_number, int):
            return particles_number
        raise TypeError(
            'Conflict between `init_type` and `particles_number`.'
        )

    @staticmethod
    def get_potential_table(potential_type: str):
        if potential_type == 'lennard_jones':
            return LennardJones().potential_table
        elif potential_type == 'dzugutov':
            return Dzugutov().potential_table
