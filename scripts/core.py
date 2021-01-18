from copy import deepcopy
from json import load
from datetime import datetime
from os.path import join
from typing import Optional

import numpy as np
import pandas as pd

from scripts.constants import PATH_TO_CONFIG, PATH_TO_DATA
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import get_parameters_dict, print_info
from scripts.isotherm import Isotherm
from scripts.log_config import log_debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters
from scripts.potential_parameters import PotentialParameters
from scripts.saver import Saver
from scripts.static_parameters import SystemStaticParameters
from scripts.verlet import Verlet


class MolecularDynamics:

    def __init__(
            self,
            config_filename: Optional[str] = None,
            is_initially_frozen: bool = True,
            is_rdf_calculated: bool = True,
            is_msd_calculated: bool = True,
    ):
        _config_filename = join(
            PATH_TO_CONFIG,
            config_filename or 'config.json'
        )
        with open(_config_filename, encoding='utf8') as file:
            config_parameters = load(file)

        self.model = ModelingParameters(**config_parameters['modeling_parameters'])
        self.static = SystemStaticParameters(**config_parameters['static_parameters'])
        if 'file_name' in config_parameters['static_parameters']:
            _file_name = join(
                PATH_TO_DATA,
                config_parameters['static_parameters']['file_name'],
            )
            configuration = pd.read_csv(_file_name, sep=';')
            self.static.cell_dimensions = configuration.loc[0, ['L_x', 'L_y', 'L_z']].to_numpy()
            self.static.particles_number = configuration.loc[0, 'particles_number']
            self.dynamic = SystemDynamicParameters(
                static=self.static,
            )
            self.dynamic.positions = configuration[['x', 'y', 'z']].to_numpy()
            self.dynamic.velocities = configuration[['v_x', 'v_y', 'v_z']].to_numpy()
            self.dynamic.accelerations = configuration[['a_x', 'a_y', 'a_z']].to_numpy()
            self.model.time = configuration.loc[0, 'time']
        else:
            self.dynamic = SystemDynamicParameters(
                static=self.static,
                temperature=self.model.initial_temperature if not is_initially_frozen else None,
            )
        self.potential = PotentialParameters(**config_parameters['potential_parameters'])
        external = ExternalParameters(**config_parameters['external_parameters'])
        attributes = {
            'static': self.static,
            'dynamic': self.dynamic,
            'model': self.model,
        }
        self.verlet = Verlet(
            external=external,
            potential=self.potential,
            **attributes,
        )
        self.saver = Saver(
            **attributes,
            **config_parameters['saver_parameters'],
        )
        self.isotherm_parameters = config_parameters['isotherm_parameters']
        self.is_rdf_calculated = is_rdf_calculated
        self.is_msd_calculated = is_msd_calculated
        self.environment_type = external.environment_type

    def md_time_step(
            self,
            potential_table: np.ndarray,
            step: int,
            system_parameters: dict = None,
            is_rdf_calculation: bool = False,
            is_pbc_switched_on: bool = False,
    ):
        system_kinetic_energy, temperature = self.verlet.system_dynamics(
            stage_id=1,
            environment_type=self.environment_type,
        )
        # self.dynamic.calculate_interparticle_vectors()
        if is_pbc_switched_on:
            self.dynamic.boundary_conditions()
        potential_energy, virial = self.verlet.load_forces(
            potential_table=potential_table,
        )
        parameters = {
            'system_kinetic_energy': system_kinetic_energy,
            'potential_energy': potential_energy,
            'virial': virial,
        }
        pressure, total_energy = self.verlet.system_dynamics(
            stage_id=2,
            environment_type=self.environment_type,
            **parameters,
        )
        parameters.update({
            'system_kinetic_energy': self.dynamic.system_kinetic_energy,
            'temperature': self.dynamic.temperature(
                system_kinetic_energy=parameters['system_kinetic_energy']
            ),
            'pressure': pressure,
            'total_energy': total_energy,
        })

        log_debug_info(f'Kinetic Energy after system_dynamics_2: {self.dynamic.system_kinetic_energy}')
        log_debug_info(f'Temperature after system_dynamics_2: {parameters["temperature"]}')
        log_debug_info(f'Pressure after system_dynamics_2: {pressure}')
        log_debug_info(f'Potential energy after system_dynamics_2: {potential_energy}')
        log_debug_info(f'Total energy after system_dynamics_2: {total_energy}')
        log_debug_info(f'Virial after system_dynamics_2: {virial}')
        if not is_rdf_calculation and system_parameters is not None:
            msd = self.dynamic.get_msd(
                previous_positions=self.dynamic.first_positions,
            )
            diffusion = msd / 6 / self.model.time_step / step
            parameters['msd'] = msd
            parameters['diffusion'] = diffusion
            log_debug_info(f'MSD after system_dynamics_2: {msd}')
            log_debug_info(f'Diffusion after system_dynamics_2: {diffusion}')

            self.saver.dynamic = self.dynamic
            self.saver.step = step
            self.saver.model.time = self.model.time
            self.saver.update_system_parameters(
                system_parameters=system_parameters,
                parameters=parameters,
            )
            self.saver.store_configuration()
            self.saver.save_configurations()
        return virial

    def fix_current_temperature(self):
        self.verlet.external.temperature = round(self.dynamic.temperature(), 5)
        if self.verlet.external.temperature == 0:
            self.verlet.external.temperature = self.model.initial_temperature

    def reduce_transition_processes(
            self,
            skipped_iterations: int = 50,
    ):
        print('Reducing Transition Processes.')
        log_debug_info('Reducing Transition Processes.')
        external_temperature = self.verlet.external.temperature
        self.fix_current_temperature()
        for _ in range(skipped_iterations):
            self.md_time_step(
                potential_table=self.potential.potential_table,
                step=1,
                is_rdf_calculation=True,
            )
        self.verlet.external.temperature = external_temperature

    def fix_external_conditions(self, virial: float = None):
        print(f'********Isotherm for T = {self.dynamic.temperature():.5f}********')
        self.fix_current_temperature()
        self.verlet.external.pressure = self.dynamic.get_pressure(
            virial=virial,
            temperature=self.verlet.external.temperature,
        )
        log_debug_info(f'External Temperature: {self.verlet.external.temperature}')
        log_debug_info(f'External Pressure: {self.verlet.external.pressure}')

    def equilibrate_system(self, equilibration_steps: int):
        virial = 0
        for eq_step in range(equilibration_steps):
            temperature = self.dynamic.temperature()
            pressure = self.dynamic.get_pressure(
                virial=virial,
                temperature=temperature,
                cell_volume=self.static.get_cell_volume(),
                density=self.static.get_density()
            )
            message = (
                f'Equilibration Step: {eq_step:3d}/{equilibration_steps}, \t'
                f'Temperature = {temperature:8.5f} epsilon/k_B, \t'
                f'Pressure = {pressure:.5f} epsilon/sigma^3, \t'
            )
            log_debug_info(message)
            print(message)
            virial = self.md_time_step(
                potential_table=self.potential.potential_table,
                step=1,
                is_rdf_calculation=True,
            )

    @logger_wraps()
    def run_md(self):
        start = datetime.now()
        system_parameters = get_parameters_dict(
            names=(
                'temperature',
                'pressure',
                'system_kinetic_energy',
                'potential_energy',
                'total_energy',
                'virial',
                'msd',
                'diffusion',
            ),
            value_size=self.model.iterations_numbers,
        )

        self.reduce_transition_processes()
        self.dynamic.first_positions = deepcopy(self.dynamic.positions)

        for step in range(1, self.model.iterations_numbers + 1):
            self.model.time += self.model.time_step
            log_debug_info(f'Step: {step}; Time: {self.model.time:.3f};')
            self.md_time_step(
                potential_table=self.potential.potential_table,
                step=step,
                system_parameters=system_parameters,
            )
            print_info(
                step=step,
                iterations_numbers=self.model.iterations_numbers,
                current_time=self.model.time,
                parameters=system_parameters,
            )
            log_debug_info(f'End of step {step}.\n')

            if self.is_rdf_calculated:
                if step in (1, 1000) or step % self.isotherm_parameters['isotherm_saving_step'] == 0:
                    Isotherm(
                        sample=deepcopy(self),
                        virial=system_parameters['virial'][step - 1],
                    ).run()

        self.saver.save_configurations(
            is_last_step=True,
        )
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
        )
        self.saver.save_configuration()
        print(f'Calculation completed. Time of calculation: {datetime.now() - start}')
