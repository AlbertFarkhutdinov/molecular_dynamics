from datetime import datetime
from os.path import join

import numpy as np
from pandas import DataFrame

from constants import PATH_TO_DATA
from dynamic_parameters import SystemDynamicParameters
from modeling_parameters import ModelingParameters
from helpers import get_empty_vectors, get_formatted_time
from static_parameters import SystemStaticParameters


class Saver:

    def __init__(
            self,
            configuration_storing_step: int = 20,
            configuration_saving_step: int = 1000,
            step: int = 1,
            static: SystemStaticParameters = None,
            dynamic: SystemDynamicParameters = None,
            model: ModelingParameters = None,
            lammps_configurations=None,
    ):
        self.step = step
        self.static = static
        self.dynamic = dynamic
        self.model = model
        self.lammps_configurations = lammps_configurations or []
        self.configuration_storing_step = configuration_storing_step
        self.configuration_saving_step = configuration_saving_step

    def save_configuration(self, file_name: str = None):
        file_name = join(
            PATH_TO_DATA,
            file_name or f'system_config_{get_formatted_time()}.txt'
        )
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

    def load_saved_configuration(self, file_name: str = None):
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

    def update_system_parameters(
            self,
            system_parameters: dict,
            potential_energy: float,
            temperature: float,
            pressure: float,
            system_kinetic_energy: float,
    ):
        system_parameters['temperature'][self.step - 1] = temperature
        system_parameters['pressure'][self.step - 1] = pressure
        system_parameters['kinetic_energy'][self.step - 1] = system_kinetic_energy
        system_parameters['potential_energy'][self.step - 1] = potential_energy
        system_parameters['total_energy'][self.step - 1] = system_kinetic_energy + potential_energy

    def get_lammps_trajectory(self):
        lines = [
            'ITEM: TIMESTEP',
            str(f'{self.model.time:.5f}'),
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

    def store_configuration(self):
        if self.step % self.configuration_storing_step == 0:
            self.lammps_configurations.append(
                self.get_lammps_trajectory()
            )

    def save_configurations(
            self,
            file_name: str = None,
            is_last_step: bool = False,
    ):
        _start = datetime.now()
        file_name = join(
            PATH_TO_DATA,
            file_name or f'system_config.txt'
        )
        is_saved = False
        _saving_step = self.configuration_saving_step
        if not is_last_step and self.step % self.configuration_saving_step == 0:
            is_saved = True
        elif is_last_step:
            _saving_step = self.model.iterations_numbers % self.configuration_saving_step
            is_saved = True
        if is_saved:
            with open(file_name, mode='a', encoding='utf-8') as file:
                file.write('\n'.join(self.lammps_configurations))
            print(
                f'LAMMPS trajectories for last {_saving_step} steps are saved. '
                f'Time of saving: {datetime.now() - _start}'
            )
            self.lammps_configurations = []

    @staticmethod
    def save_dict(
            data: dict,
            default_file_name: str,
            data_name: str,
            file_name: str = None,
    ):
        _start = datetime.now()
        _file_name = join(PATH_TO_DATA, file_name or default_file_name)
        DataFrame(data).to_csv(
            _file_name,
            sep=';',
            index=False,
        )
        print(f'{data_name} are saved. Time of saving: {datetime.now() - _start}')

    def save_system_parameters(
            self,
            system_parameters: dict,
            file_name: str = None,
    ):
        self.save_dict(
            data=system_parameters,
            default_file_name=f'system_parameters.csv',
            data_name='System parameters',
            file_name=file_name,
        )

    def save_rdf(
            self,
            rdf_data,
            file_name: str = None,
    ):
        self.save_dict(
            data=rdf_data,
            default_file_name=f'rdf_file_{get_formatted_time()}.csv',
            data_name='RDF values',
            file_name=file_name,
        )
