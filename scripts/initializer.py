import numpy as np

from scripts.immutable_parameters import ImmutableParameters
from scripts.helpers import get_empty_vectors
from scripts.log_config import logger_wraps


class Initializer:

    def __init__(
            self,
            immutables: ImmutableParameters,
            **initial_parameters,
    ):
        self.immutables = immutables
        self.init_type = immutables.init_type
        self.cell_dimensions = np.zeros(3, dtype=np.float)
        self.particles_number = immutables.particles_number
        self.positions = get_empty_vectors(self.particles_number)
        self.velocities = get_empty_vectors(self.particles_number)
        self.accelerations = get_empty_vectors(self.particles_number)
        self.virial = 0.0
        self.potential_energy = 0.0
        if self.immutables.init_type == 1:
            self.generate_ordered_state()
        elif self.immutables.init_type == 2:
            self.generate_random_state()
        if initial_parameters.get('temperature'):
            self.get_initial_velocities(
                temperature=initial_parameters['temperature']
            )

    def get_cell_dimensions(self):
        if self.immutables.init_type == -1:
            self.cell_dimensions = np.zeros(3, dtype=np.float)
        elif self.immutables.init_type in (1, 2):
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
                self.cell_dimensions = np.array(particles_number, dtype=np.float) * self.lattice_constant
            elif self.init_type == 2 and isinstance(particles_number, int):
                self.cell_dimensions = (
                        np.ones(3, dtype=np.float)
                        * self.lattice_constant
                        * np.round((particles_number / self.nb) ** (1/3))
                )
            else:
                raise TypeError('Conflict between `init_type` and `particles_number`.')
        else:
            raise ValueError('Unacceptable `init_type`.')

    @logger_wraps()
    def generate_ordered_state(self) -> None:
        r_cell = get_empty_vectors(self.immutables.nb)
        if self.immutables.crystal_type == 'гцк':
            r_cell[1][0] = 0.5
            r_cell[3][0] = 0.5
            r_cell[1][1] = 0.5
            r_cell[2][1] = 0.5
            r_cell[2][2] = 0.5
            r_cell[3][2] = 0.5
        n = 0
        for k in range(self.immutables.particles_numbers[2]):
            for j in range(self.immutables.particles_numbers[1]):
                for i in range(self.immutables.particles_numbers[0]):
                    for ll in range(self.immutables.nb):
                        n += 1
                        self.positions[n - 1] = (
                                self.immutables.lattice_constant *
                                (np.array([i, j, k]) + r_cell[ll])
                        )
        self.load_system_center()

    @logger_wraps()
    def generate_random_state(self) -> None:
        # TODO raises AssertionError
        np.random.seed(0)
        self.positions[0] = np.random.random(3) * self.immutables.cell_dimensions
        for j in range(1, self.immutables.particles_number):
            is_distance_too_small = True
            while is_distance_too_small:
                self.positions[j] = np.random.random(3) * self.immutables.cell_dimensions
                for i in range(j):
                    radius_vector = self.positions[i] - self.positions[j]
                    radius_vector -= (radius_vector / self.cell_dimensions).astype(np.int32) * self.cell_dimensions
                    distance = (radius_vector ** 2).sum()
                    is_distance_too_small = (distance < 1.1)
        self.load_system_center()

    @property
    def system_center(self):
        return self.positions.sum(axis=0) / self.particles_number

    @logger_wraps()
    def load_system_center(self) -> None:
        self.positions -= self.system_center

    def get_initial_velocities(self, temperature: float) -> None:
        np.random.seed(0)
        _sigma = np.sqrt(temperature)
        velocities = get_empty_vectors(self.particles_number).transpose()
        for i in range(3):
            velocities[i] = np.random.normal(0.0, _sigma, self.particles_number)
            velocities[i] -= velocities[i].sum() / self.particles_number

        scale_factor = np.sqrt(3.0 * temperature * self.particles_number / (velocities * velocities).sum())
        for i in range(3):
            velocities[i] *= scale_factor
        self.velocities = velocities.transpose()
