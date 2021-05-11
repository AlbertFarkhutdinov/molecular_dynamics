from os.path import join
from typing import Iterable, Optional

import numpy as np
from numpy.random import random
import pandas as pd

from scripts.constants import PATH_TO_DATA
from scripts.helpers import get_empty_vectors
from scripts.log_config import logger_wraps
from scripts.numba_procedures import check_distances
from scripts.system import Configuration, System


class Initializer:

    def __init__(
            self,
            system: System,
            **initial_parameters,
    ):
        self.system = system
        self.init_type = initial_parameters.get("init_type")
        if self.init_type == -1:
            self.initialize_from_file(
                file_name=initial_parameters.get("file_name"),
            )
        elif self.init_type == 1:
            self.initialize_as_crystal(
                crystal_type=initial_parameters.get("crystal_type"),
                lattice_constant=initial_parameters.get("lattice_constant"),
                particles_numbers=initial_parameters.get("particles_numbers"),
            )
        elif self.init_type == 2:
            self.initialize_random_configuration(
                cell_dimensions=initial_parameters.get("cell_dimensions"),
                particles_number=initial_parameters.get("particles_number"),
            )
        else:
            raise ValueError('Unacceptable `init_type`.')
        temperature = initial_parameters.get("initial_temperature", 0.0)
        if temperature:
            self.get_initial_velocities(temperature=temperature)
            self.system.configuration.temperature = temperature
        self.system.pressure = initial_parameters.get(
            "initial_pressure",
            self.system.get_pressure(temperature=temperature)
        )

    def initialize_from_file(self, file_name: str):
        _file_name = join(PATH_TO_DATA, file_name)
        _configuration = pd.read_csv(_file_name, sep=';')
        self.system.cell_dimensions = _configuration.loc[
            0,
            ['L_x', 'L_y', 'L_z'],
        ].to_numpy()
        self.system.initial_cell_dimensions = _configuration.loc[
            0,
            ['L_x', 'L_y', 'L_z'],
        ].to_numpy()
        self.system.configuration.particles_number = _configuration.loc[
            0,
            'particles_number',
        ]
        self.system.configuration.positions = _configuration[
            ['x', 'y', 'z']
        ].to_numpy()
        self.system.configuration.velocities = _configuration[
            ['v_x', 'v_y', 'v_z']
        ].to_numpy()
        self.system.configuration.accelerations = _configuration[
            ['a_x', 'a_y', 'a_z']
        ].to_numpy()
        self.system.time = _configuration.loc[0, 'time']

    @staticmethod
    def get_lattice_points(crystal_type: str) -> Optional[int]:
        _lattice_points_numbers = {
            'PC': 1,
            'BCC': 2,
            'FCC': 4,
        }
        if crystal_type in _lattice_points_numbers:
            return _lattice_points_numbers[crystal_type]
        raise KeyError('Unacceptable `crystal_type`.')

    def initialize_configuration_with_particles_number(
            self,
            particles_number: int,
    ):
        self.system.configuration.positions = get_empty_vectors(
            particles_number
        )
        self.system.configuration.velocities = get_empty_vectors(
            particles_number
        )
        self.system.configuration.accelerations = get_empty_vectors(
            particles_number
        )

    def initialize_as_crystal(
            self,
            crystal_type: str,
            lattice_constant: float,
            particles_numbers: Iterable,
    ):
        lattice_points = self.get_lattice_points(crystal_type=crystal_type)
        self.system.configuration.particles_number = (
                lattice_points
                * particles_numbers[0]
                * particles_numbers[1]
                * particles_numbers[2]
        )
        self.system.cell_dimensions = np.array(
            particles_numbers,
            dtype=np.float,
        ) * lattice_constant

        self.system.initial_cell_dimensions = np.array(
            particles_numbers,
            dtype=np.float,
        ) * lattice_constant

        self.initialize_configuration_with_particles_number(
            self.system.configuration.particles_number
        )
        self.generate_ordered_state(
            crystal_type=crystal_type,
            lattice_constant=lattice_constant,
            lattice_points=lattice_points,
            particles_numbers=particles_numbers,
        )

    def initialize_random_configuration(
            self,
            particles_number: int,
            cell_dimensions: Iterable,
    ):
        self.system.configuration.particles_number = particles_number
        self.system.cell_dimensions = np.array(cell_dimensions, dtype=np.float)
        self.system.initial_cell_dimensions = np.array(
            cell_dimensions,
            dtype=np.float,
        )
        self.initialize_configuration_with_particles_number(
            self.system.configuration.particles_number
        )
        self.generate_random_state()

    @logger_wraps()
    def generate_ordered_state(
            self,
            crystal_type: str,
            lattice_points: int,
            lattice_constant: float,
            particles_numbers: Iterable,
    ) -> None:
        r_cell = get_empty_vectors(lattice_points)
        if crystal_type:
            r_cell[1][0] = 0.5
            r_cell[3][0] = 0.5
            r_cell[1][1] = 0.5
            r_cell[2][1] = 0.5
            r_cell[2][2] = 0.5
            r_cell[3][2] = 0.5
        position_number = 0
        for k in range(particles_numbers[2]):
            for j in range(particles_numbers[1]):
                for i in range(particles_numbers[0]):
                    for point in range(lattice_points):
                        position_number += 1
                        self.system.configuration.positions[
                            position_number - 1
                        ] = (
                                lattice_constant *
                                (np.array([i, j, k]) + r_cell[point])
                        )
        self.load_system_center()

    @logger_wraps()
    def generate_random_state(self) -> None:
        np.random.seed(0)
        self.system.configuration.positions[0] = (
                random(3)
                * self.system.cell_dimensions
        )
        for j in range(1, self.system.configuration.particles_number):
            is_distance_too_small = True
            while is_distance_too_small:
                self.system.configuration.positions[j] = (
                        random(3) * self.system.cell_dimensions
                )
                is_distance_too_small = check_distances(
                    particle_index=j,
                    positions=self.system.configuration.positions,
                    cell_dimensions=self.system.cell_dimensions,
                )
        self.load_system_center()

    @logger_wraps()
    def load_system_center(self) -> None:
        self.system.configuration.positions -= (
            self.system.configuration.get_system_center()
        )

    def get_initial_velocities(self, temperature: float) -> None:
        np.random.seed(0)
        _sigma = np.sqrt(temperature)
        _particles_number = self.system.configuration.particles_number
        velocities = get_empty_vectors(_particles_number).transpose()
        for i in range(3):
            velocities[i] = np.random.normal(
                0.0,
                _sigma,
                _particles_number,
            )
            velocities[i] -= velocities[i].sum() / _particles_number

        scale_factor = (
            np.sqrt(
                3.0 * temperature * _particles_number
                / (velocities * velocities).sum()
            )
        )
        for i in range(3):
            velocities[i] *= scale_factor
        self.system.configuration.velocities = velocities.transpose()

    def get_initials(self) -> System:
        system = System(
            configuration=Configuration(
                particles_number=self.system.configuration.particles_number,
                positions=self.system.configuration.positions,
                velocities=self.system.configuration.velocities,
                accelerations=self.system.configuration.accelerations,
                temperature=self.system.configuration.temperature,
            ),
            cell_dimensions=self.system.cell_dimensions,
            pressure=self.system.pressure,
        )
        volume = system.volume
        system.get_density(volume=volume)
        return system
