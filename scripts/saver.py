from datetime import datetime
from os import mkdir
from os.path import exists, join

import pandas as pd

from scripts.constants import PATH_TO_DATA
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.modeling_parameters import ModelingParameters
from scripts.helpers import get_formatted_time, get_date
from scripts.static_parameters import SystemStaticParameters


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
        self.date_folder = join(
            PATH_TO_DATA,
            get_date(),
        )
        if not exists(self.date_folder):
            mkdir(self.date_folder)

    def save_configuration(self, file_name: str = None):
        _start = datetime.now()
        _file_name = join(
            self.date_folder,
            file_name or 'system_configuration.csv',
        )
        positions = pd.DataFrame(
            self.dynamic.positions,
            columns=['x', 'y', 'z'],
        )
        velocities = pd.DataFrame(
            self.dynamic.velocities,
            columns=['v_x', 'v_y', 'v_z'],
        )
        accelerations = pd.DataFrame(
            self.dynamic.accelerations,
            columns=['a_x', 'a_y', 'a_z'],
        )
        configuration = pd.concat([positions, velocities, accelerations], axis=1)
        configuration[['L_x', 'L_y', 'L_z']] = self.static.cell_dimensions
        configuration['particles_number'] = self.static.particles_number
        configuration['time'] = self.model.time

        configuration.to_csv(
            _file_name,
            sep=';',
            index=False,
        )
        print(f'System configuration is saved. Time of saving: {datetime.now() - _start}')

    def update_system_parameters(
            self,
            system_parameters: dict,
            parameters: dict,
    ):
        for key, value in parameters.items():
            system_parameters[key][self.step - 1] = value

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
            self.date_folder,
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

    def save_dict(
            self,
            data: dict,
            default_file_name: str,
            data_name: str,
            file_name: str = None,
    ):
        _start = datetime.now()
        _file_name = join(
            self.date_folder,
            file_name or default_file_name,
        )
        pd.DataFrame(data).to_csv(
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
