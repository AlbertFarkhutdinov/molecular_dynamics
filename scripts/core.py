from copy import deepcopy
# from cProfile import run
from json import load
from math import pi
from datetime import datetime
from os.path import join
from time import time
from typing import Optional

import numpy as np
import pandas as pd

from scripts.constants import PATH_TO_CONFIG, PATH_TO_DATA
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import get_empty_float_scalars, get_parameters_dict, print_info
from scripts.log_config import log_debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters
from scripts.numba_procedures import get_interparticle_distances, get_time_displacements
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

    def run_isotherm(
            self,
            virial: float = None,
            layer_thickness: float = 0.01,
    ):
        start = time()
        sample = deepcopy(self)
        sample.fix_external_conditions(
            virial=virial,
        )
        sample.equilibrate_system(
            equilibration_steps=sample.isotherm_parameters['equilibration_steps'],
        )

        ensembles_number = sample.isotherm_parameters['ensembles_number']
        steps_number = 2 * ensembles_number - 1
        isotherm_system_parameters = get_parameters_dict(
            names=(
                'time',
                'msd',
                'einstein_diffusion',
                'velocity_autocorrelation',
                'green_kubo_diffusion',
            ),
            value_size=steps_number,
        )

        first_positions, first_velocities = {}, {}
        rdf = get_empty_float_scalars(20 * sample.static.particles_number)
        # van_hove = np.array([
        #     get_empty_float_scalars(20 * sample.static.particles_number) for _ in range(ensembles_number)
        # ])
        green_kubo_diffusion = 0

        print(f'********RDF Calculation started********')
        for rdf_step in range(1, steps_number + 1):
            temperature = sample.dynamic.temperature()
            pressure = sample.dynamic.get_pressure(
                virial=virial,
                temperature=temperature,
                cell_volume=sample.static.get_cell_volume(),
                density=sample.static.get_density()
            )
            message = (
                f'RDF Step: {rdf_step}/{steps_number}, '
                f'Temperature = {temperature:8.5f} epsilon/k_B, \t'
                f'Pressure = {pressure:.5f} epsilon/sigma^3, \t'
            )
            log_debug_info(message)
            print(message)

            virial = sample.md_time_step(
                potential_table=sample.potential.potential_table,
                step=rdf_step,
                is_rdf_calculation=True,
            )

            first_step = rdf_step - ensembles_number
            if rdf_step <= ensembles_number:
                first_step = 0
                first_positions[rdf_step] = deepcopy(sample.dynamic.positions)
                first_velocities[rdf_step] = deepcopy(sample.dynamic.velocities)
                # static_distances = deepcopy(sample.dynamic.interparticle_distances)
                sample.dynamic.calculate_interparticle_vectors()
                static_radius_vectors = sample.dynamic.interparticle_vectors
                static_distances = sample.dynamic.interparticle_distances
                # static_radius_vectors
                # static_radius_vectors, static_distances
                # static_distances = get_interparticle_distances(
                #     distances=np.zeros(
                #         (sample.static.particles_number, sample.static.particles_number),
                #         dtype=np.float,
                #     ),
                #     positions=sample.dynamic.positions,
                #     cell_dimensions=sample.static.cell_dimensions,
                # )

                # TODO реализовать статический и динамический структурный фактор, функцию Ван-Хова, рассеяния

                radiuses = np.arange(layer_thickness, static_distances.max() + 1, layer_thickness)
                rdf_hist = np.histogram(
                    static_distances.flatten(),
                    radiuses
                )[0]
                rdf[:rdf_hist.size] += (
                        2.0 * sample.static.get_cell_volume()
                        / (4.0 * pi * radiuses[:rdf_hist.size] ** 2
                           * sample.static.particles_number * sample.static.particles_number)
                        * rdf_hist / layer_thickness
                )
                isotherm_system_parameters['time'][rdf_step - 1] = sample.model.time_step * rdf_step

            for i in range(first_step, rdf_step):
                isotherm_system_parameters['msd'][i] += sample.dynamic.get_msd(
                    previous_positions=first_positions[rdf_step - i],
                )
                isotherm_system_parameters['velocity_autocorrelation'][i] += (
                    (first_velocities[rdf_step - i] * sample.dynamic.velocities).sum()
                    / sample.static.particles_number
                )
                # dynamic_distances = get_time_displacements(
                #     positions_1=sample.dynamic.positions,
                #     positions_2=first_positions[rdf_step - i],
                #     distances=np.zeros(
                #         (sample.static.particles_number, sample.static.particles_number),
                #         dtype=np.float,
                #     ),
                # )
                # radiuses = np.arange(layer_thickness, dynamic_distances.max() + 1, layer_thickness)
                # van_hove_hist = np.histogram(
                #     dynamic_distances.flatten(),
                #     radiuses
                # )[0]
                # van_hove[i][:van_hove_hist.size] += (
                #         sample.static.get_cell_volume()
                #         / (2.0 * pi * radiuses[:van_hove_hist.size] ** 2
                #            * sample.static.particles_number * sample.static.particles_number)
                #         * van_hove_hist / layer_thickness
                # )

        for key, value in isotherm_system_parameters.items():
            isotherm_system_parameters[key] = value[:ensembles_number]

        isotherm_system_parameters['msd'] = isotherm_system_parameters['msd'] / ensembles_number
        isotherm_system_parameters[
            'velocity_autocorrelation'
        ] = isotherm_system_parameters['velocity_autocorrelation'] / ensembles_number

        isotherm_system_parameters[
            'einstein_diffusion'
        ] = isotherm_system_parameters['msd'] / 6 / isotherm_system_parameters['time']

        for i in range(ensembles_number):
            green_kubo_diffusion += isotherm_system_parameters[
                                        'velocity_autocorrelation'
                                    ][i] * sample.model.time_step / 3
            isotherm_system_parameters['green_kubo_diffusion'][i] += green_kubo_diffusion

        rdf = rdf[:np.nonzero(rdf)[0][-1]] / ensembles_number
        radiuses = layer_thickness * np.arange(1, rdf.size + 1)
        Saver().save_rdf(
            rdf_data={
                'radius': radiuses[radiuses <= sample.static.cell_dimensions[0] / 2.0],
                'rdf': rdf[radiuses <= sample.static.cell_dimensions[0] / 2.0],
            },
            file_name=f'rdf_T_{sample.verlet.external.temperature:.5f}.csv'
        )
        Saver().save_dict(
            data=isotherm_system_parameters,
            default_file_name=f'transport.csv',
            data_name='MSD and self-diffusion coefficient',
            file_name=f'transport_T_{sample.verlet.external.temperature:.5f}.csv'
        )
        print(f'Calculation completed. Time of calculation: {time() - start} seconds.')

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

        # self.reduce_transition_processes()
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
                    self.run_isotherm(virial=system_parameters['virial'][step - 1])

        self.saver.save_configurations(
            is_last_step=True,
        )
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
        )
        self.saver.save_configuration()
        print(f'Calculation completed. Time of calculation: {datetime.now() - start}')
