import numpy as np
import numba
from scipy.spatial.distance import pdist, squareform

from static_parameters import SystemStaticParameters
from helpers import get_empty_vectors


class SystemDynamicParameters:

    def __init__(
            self,
            static: SystemStaticParameters,
    ):
        self.static = static
        self.particles_number = static.particles_number
        self.cell_dimensions = static.cell_dimensions
        self.positions = get_empty_vectors(self.particles_number)
        self.velocities = get_empty_vectors(self.particles_number)
        self.accelerations = get_empty_vectors(self.particles_number)
        if self.static.init_type == 1:
            self.generate_ordered_state()
        elif self.static.init_type == 2:
            self.generate_random_state()

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

    def generate_random_state(self) -> None:
        self.positions[0] = np.random.random(3) * self.static.cell_dimensions
        for j in range(1, self.static.particles_number):
            is_distance_too_small = True
            while is_distance_too_small:
                self.positions[j] = np.random.random(3) * self.static.cell_dimensions
                for i in range(j):
                    distance = self.get_distance(i, j)
                    is_distance_too_small = (distance < 1.1)
        self.load_system_center()

    @property
    def system_kinetic_energy(self) -> float:
        return (self.velocities ** 2).sum() / 2.0

    @property
    def temperature(self) -> float:
        return 2.0 * self.system_kinetic_energy / 3.0 / self.particles_number

    @property
    def interparticle_distances(self):
        return squareform(pdist(self.positions, 'euclidean'))

    def get_pressure(
            self,
            virial: float,
            temperature: float = None,
            density: float = None,
            cell_volume: float = None,
    ) -> float:
        _density = density or self.static.get_density()
        _temperature = temperature or self.temperature
        _cell_volume = cell_volume or self.static.get_cell_volume()
        return _density * _temperature + virial / (3 * _cell_volume)

    def load_system_center(self) -> None:
        system_center = self.positions.sum(axis=0)
        system_center = system_center / self.particles_number
        self.positions -= system_center

    def get_next_positions(
            self,
            time_step: float,
            acc_coefficient=0,
    ) -> None:
        self.positions += (
                self.velocities * time_step
                + (self.accelerations - acc_coefficient * self.velocities)
                * (time_step ** 2) / 2.0
        )

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

    def boundary_conditions(self) -> None:
        self.positions -= self.cell_dimensions * (
                self.positions >= self.cell_dimensions / 2.0
        ).astype(np.float)
        self.positions += self.cell_dimensions * (
                self.positions < -self.cell_dimensions / 2.0
        ).astype(np.float)

    def get_radius_vector(self, index_1: int, index_2):
        return self.positions[index_1] - self.positions[index_2]
        # return self._get_radius_vector(
        #     pos_1=self.positions[index_1],
        #     pos_2=self.positions[index_2],
        #     # dim=self.cell_dimensions,
        # )

    # @staticmethod
    # @numba.njit(numba.float64[:](numba.float64[:], numba.float64[:]))
    # def _get_radius_vector(
    #         pos_1,
    #         pos_2,
    #         # dim,
    # ):
    #     r_ij = pos_1 - pos_2
    #     # _r_ij = r_ij
    #     # r_ij -= (
    #     #         (r_ij / dim).astype(np.int32)
    #     #         * dim
    #     # )
    #     # if (r_ij - _r_ij).any():
    #     #     print(_r_ij, r_ij)
    #     return r_ij

    @staticmethod
    @numba.njit(numba.float64(numba.float64[:]))
    def _get_distance(radius_vector):
        return np.linalg.norm(radius_vector)

    def get_distance(self, index_1: int, index_2: int, radius_vector: np.ndarray = None):
        if radius_vector is None:
            radius_vector = self.get_radius_vector(index_1, index_2)
        return self._get_distance(radius_vector)
