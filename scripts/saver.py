from os import mkdir
from os.path import exists, join

import pandas as pd

from scripts.constants import PATH_TO_DATA
from scripts.helpers import get_formatted_time, get_current_time, get_date
from scripts.simulation_parameters import SimulationParameters
from scripts.system import System


class Saver:

    def __init__(
            self,
            simulation_parameters: SimulationParameters = None,
            step: int = 1,
            system: System = None,
            lammps_configurations = None,
            parameters_saving_step: int = None,

    ):
        self.step = step
        self.system = system
        self.iterations_numbers = simulation_parameters.iterations_numbers
        self.parameters_saving_step = (
                parameters_saving_step or self.iterations_numbers
        )
        self.lammps_configurations = lammps_configurations or []
        self.configuration_storing_step = (
            simulation_parameters.configuration_storing_step
        )
        self.configuration_saving_step = (
            simulation_parameters.configuration_saving_step
        )

    @property
    def date_folder(self):
        _date_folder = join(PATH_TO_DATA, get_date())
        if not exists(_date_folder):
            mkdir(_date_folder)
        return _date_folder

    def save_configuration(self, file_name: str = None):
        _start = get_current_time()
        _file_name = join(
            self.date_folder,
            file_name or 'system_configuration.csv',
        )
        positions = pd.DataFrame(
            self.system.configuration.positions,
            columns=['x', 'y', 'z'],
        )
        velocities = pd.DataFrame(
            self.system.configuration.velocities,
            columns=['v_x', 'v_y', 'v_z'],
        )
        accelerations = pd.DataFrame(
            self.system.configuration.accelerations,
            columns=['a_x', 'a_y', 'a_z'],
        )
        configuration = pd.concat(
            [positions, velocities, accelerations],
            axis=1,
        )
        configuration[['L_x', 'L_y', 'L_z']] = self.system.cell_dimensions
        configuration[
            'particles_number'
        ] = self.system.configuration.particles_number
        configuration['time'] = self.system.time

        configuration.to_csv(
            _file_name,
            sep=';',
            index=False,
        )
        print(
            f'The last system configuration is saved. '
            f'Time of saving: {get_current_time() - _start}'
        )

    def update_system_parameters(
            self,
            system_parameters: dict,
            parameters: dict,
    ):
        for key, value in parameters.items():
            system_parameters[key][
                self.step % self.parameters_saving_step - 1
            ] = value

    def get_lammps_trajectory(self):
        lines = [
            'ITEM: TIMESTEP',
            str(f'{self.system.time:.5f}'),
            'ITEM: NUMBER OF ATOMS',
            str(self.system.configuration.particles_number),
            'ITEM: BOX BOUNDS pp pp pp',
            *(
                f'{-dim / 2} {dim / 2}'
                for dim in self.system.cell_dimensions
            ),
            'ITEM: ATOMS id type x y z',
            *(
                f'{i + 1} 0 {pos[0]} {pos[1]} {pos[2]}'
                for i, pos in enumerate(self.system.configuration.positions)
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
        _start = get_current_time()
        file_name = join(
            self.date_folder,
            file_name or 'system_config.txt'
        )
        is_saved = False
        _saving_step = self.configuration_saving_step
        if (
                not is_last_step
                and self.step % self.configuration_saving_step == 0
        ):
            is_saved = True
        elif is_last_step:
            _saving_step = (
                    self.iterations_numbers
                    % self.configuration_saving_step
            )
            is_saved = True
        if is_saved:
            with open(file_name, mode='a', encoding='utf-8') as file:
                file.write('\n'.join(self.lammps_configurations))
            print(
                f'LAMMPS trajectories for last {_saving_step} steps are saved.'
                f' Time of saving: {get_current_time() - _start}'
            )
            self.lammps_configurations = []

    def save_dict(
            self,
            data: dict,
            default_file_name: str,
            data_name: str,
            file_name: str = None,
    ):
        _start = get_current_time()
        _file_name = join(
            self.date_folder,
            file_name or default_file_name,
        )
        pd.DataFrame(data).to_csv(
            _file_name,
            sep=';',
            index=False,
        )
        print(
            f'{data_name} are saved. '
            f'Time of saving: {get_current_time() - _start}'
        )

    def save_system_parameters(
            self,
            system_parameters: dict,
            file_name: str = None,
    ):
        self.save_dict(
            data=system_parameters,
            default_file_name='system_parameters.csv',
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
            default_file_name=f'rdf_{get_formatted_time()}.csv',
            data_name='RDF values',
            file_name=file_name,
        )
