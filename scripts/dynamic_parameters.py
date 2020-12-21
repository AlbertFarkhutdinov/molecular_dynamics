import numpy as np
import numba
from scipy.spatial.distance import pdist, squareform

from scripts.static_parameters import SystemStaticParameters
from scripts.helpers import get_empty_vectors
from scripts.log_config import logger_wraps


class SystemDynamicParameters:

    def __init__(
            self,
            static: SystemStaticParameters,
            temperature: float = None,
    ):
        self.static = static
        self.particles_number = static.particles_number
        self.cell_dimensions = static.cell_dimensions
        self.positions = get_empty_vectors(self.particles_number)
        self.first_positions = get_empty_vectors(self.particles_number)
        self.first_velocities = get_empty_vectors(self.particles_number)
        self.velocities = get_empty_vectors(self.particles_number)
        self.accelerations = get_empty_vectors(self.particles_number)
        if self.static.init_type == 1:
            self.generate_ordered_state()
        elif self.static.init_type == 2:
            self.generate_random_state()
        if temperature:
            self.get_initial_velocities(temperature=temperature)
        self.displacements = get_empty_vectors(self.particles_number)

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

    @logger_wraps()
    def generate_ordered_state(self) -> None:
        r_cell = get_empty_vectors(self.static.nb)
        if self.static.crystal_type == 'гцк':
            r_cell[1][0] = 0.5
            r_cell[3][0] = 0.5
            r_cell[1][1] = 0.5
            r_cell[2][1] = 0.5
            r_cell[2][2] = 0.5
            r_cell[3][2] = 0.5
        n = 0
        for k in range(self.static.particles_numbers[2]):
            for j in range(self.static.particles_numbers[1]):
                for i in range(self.static.particles_numbers[0]):
                    for ll in range(self.static.nb):
                        n += 1
                        self.positions[n - 1] = (
                                self.static.lattice_constant *
                                (np.array([i, j, k]) + r_cell[ll])
                        )
        self.load_system_center()

    @logger_wraps()
    def generate_random_state(self) -> None:
        # TODO raises AssertionError
        np.random.seed(0)
        self.positions[0] = np.random.random(3) * self.static.cell_dimensions
        for j in range(1, self.static.particles_number):
            is_distance_too_small = True
            while is_distance_too_small:
                self.positions[j] = np.random.random(3) * self.static.cell_dimensions
                for i in range(j):
                    # radius_vector = np.mod(self.positions[i] - self.positions[j], self.cell_dimensions)
                    radius_vector = self.positions[i] - self.positions[j]
                    radius_vector -= (radius_vector / self.cell_dimensions).astype(np.int32) * self.cell_dimensions
                    distance = (radius_vector ** 2).sum()
                    is_distance_too_small = (distance < 1.1)
        self.load_system_center()

    @property
    def system_kinetic_energy(self) -> float:
        return (self.velocities * self.velocities).sum() / 2.0

    def temperature(self, system_kinetic_energy=None) -> float:
        _system_kinetic_energy = system_kinetic_energy or self.system_kinetic_energy
        return 2.0 * _system_kinetic_energy / 3.0 / self.particles_number

    @property
    def interparticle_distances(self):
        return squareform(
            pdist(self.positions, 'euclidean')
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _interparticle_distances(positions):
        return pdist(positions, 'euclidean')

    def get_pressure(
            self,
            virial: float,
            temperature: float,
            density: float = None,
            cell_volume: float = None,
    ) -> float:
        _density = density or self.static.get_density()
        _cell_volume = cell_volume or self.static.get_cell_volume()
        return _density * temperature + virial / (3 * _cell_volume)

    @property
    def system_center(self):
        return self.positions.sum(axis=0) / self.particles_number

    @logger_wraps()
    def load_system_center(self) -> None:
        self.positions -= self.system_center

    @logger_wraps()
    def get_next_positions(
            self,
            time_step: float,
            acc_coefficient=0,
    ) -> None:
        self.positions += (
                self.velocities * time_step
                + (self.accelerations - acc_coefficient * self.velocities)
                * (time_step * time_step) / 2.0
        )

    @logger_wraps()
    def get_next_velocities(
            self,
            time_step: float,
            vel_coefficient=1,
            acc_coefficient=0,
    ) -> None:
        self.velocities = (
                vel_coefficient * self.velocities
                + (self.accelerations - acc_coefficient * self.velocities)
                * time_step / 2.0
        )

    @logger_wraps()
    def boundary_conditions(self) -> None:
        self._boundary_conditions(
            cell_dimensions=self.cell_dimensions,
            particles_number=self.particles_number,
            positions=self.positions
        )

    @staticmethod
    @numba.jit(nopython=True)
    def _boundary_conditions(
            cell_dimensions: np.ndarray,
            particles_number: int,
            positions: np.ndarray,
    ):
        for i in range(particles_number):
            for j in range(3):
                if positions[i][j] >= cell_dimensions[j] / 2.0:
                    positions[i][j] -= cell_dimensions[j]
                if positions[i][j] < -cell_dimensions[j] / 2.0:
                    positions[i][j] += cell_dimensions[j]

    def distance_refold(self):
        for i in range(self.particles_number - 1):
            for j in range(i + 1, self.particles_number):
                # radius_vector = np.mod(self.get_radius_vector(i, j), self.cell_dimensions)
                radius_vector = self.get_radius_vector(i, j)
                radius_vector -= (radius_vector / self.cell_dimensions).astype(np.int) * self.cell_dimensions

    def get_radius_vector(self, index_1: int, index_2):
        return self.positions[index_1] - self.positions[index_2]

    @staticmethod
    @numba.njit(numba.float64(numba.float64[:]))
    def _get_distance(radius_vector):
        return np.linalg.norm(radius_vector)

    def get_distance(
            self,
            index_1: int,
            index_2: int,
    ):
        # radius_vector = np.mod(self.positions[index_1] - self.positions[index_2], self.cell_dimensions)
        radius_vector = self.positions[index_1] - self.positions[index_2]
        radius_vector -= (radius_vector / self.cell_dimensions).astype(numba.int32) * self.cell_dimensions
        return (radius_vector ** 2).sum()

    def get_msd(self, previous_positions):
        return ((self.positions - previous_positions) ** 2).sum() / self.particles_number

