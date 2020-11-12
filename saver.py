from os.path import join

import numpy as np

from constants import PATH_TO_DATA
from dynamic_parameters import SystemDynamicParameters
from helpers import get_empty_vectors
from modeling_parameters import ModelingParameters
from static_parameters import SystemStaticParameters


class Saver:

    def __init__(
            self,
            static: SystemStaticParameters,
            dynamic: SystemDynamicParameters,
            model: ModelingParameters,
    ):
        self.static = static
        self.dynamic = dynamic
        self.model = model

    def save_config(self, file_name: str = None):
        file_name = join(PATH_TO_DATA, file_name or 'system_config_149.txt')
        with open(file_name, mode='w', encoding='utf-8') as file:
            lines = [
                '\n'.join(self.static.cell_dimensions.astype(str)),
                str(self.static.particles_number),
                str(self.model.time),
                *[
                    f'{pos[0]} {pos[1]} {pos[2]}'
                    for i, pos in enumerate(self.dynamic.positions)
                ],
                *[
                    f'{vel[0]} {vel[1]} {vel[2]}'
                    for i, vel in enumerate(self.dynamic.velocities)
                ],
                *[
                    f'{acc[0]} {acc[1]} {acc[2]}'
                    for i, acc in enumerate(self.dynamic.accelerations)
                ],
            ]
            file.write('\n'.join(lines))

    def load_save_config(self, file_name: str = None):
        file_name = join(PATH_TO_DATA, file_name or 'system_config.txt')
        with open(file_name, mode='r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            self.static.cell_dimensions = np.array(lines[:3], dtype=np.float)
            self.static.particles_number = int(lines[3])
            self.model.time = float(lines[4])
            self.dynamic.positions = get_empty_vectors(self.static.particles_number)
            self.dynamic.velocities = get_empty_vectors(self.static.particles_number)
            self.dynamic.accelerations = get_empty_vectors(self.static.particles_number)
            for i in range(self.static.particles_number):
                self.dynamic.positions[i] = np.array(
                    lines[5 + i].split(),
                    dtype=np.float,
                )
                self.dynamic.velocities[i] = np.array(
                    lines[5 + i + self.static.particles_number].split(),
                    dtype=np.float,
                )
                self.dynamic.velocities[i] = np.array(
                    lines[5 + i + 2 * self.static.particles_number].split(),
                    dtype=np.float,
                )

    @staticmethod
    def save_system_parameters(
            system_parameters: dict,
            step: int,
            potential_energy: float,
            temperature: float,
            pressure: float,
            system_kinetic_energy: float,
    ):
        system_parameters['temperature'][step - 1] = temperature
        system_parameters['pressure'][step - 1] = pressure
        system_parameters['kinetic_energy'][step - 1] = system_kinetic_energy
        system_parameters['potential_energy'][step - 1] = potential_energy
        system_parameters['total_energy'][step - 1] = system_kinetic_energy + potential_energy

    def load_lammps_trajectory(self):
        lines = [
            'ITEM: TIMESTEP',
            str(self.model.time),
            'ITEM: NUMBER OF ATOMS',
            str(self.static.particles_number),
            'ITEM: BOX BOUNDS pp pp pp',
            *(
                f'{-dim / 2} {dim / 2}'
                for dim in self.static.cell_dimensions
            ),
            'ITEM: ATOMS id type x y z',
            *(
                f'{i + 1} 0 {pos[0]} {pos[1]} {pos[2]}'
                for i, pos in enumerate(self.dynamic.positions)
            ),
            '\n',
        ]
        return '\n'.join(lines)
