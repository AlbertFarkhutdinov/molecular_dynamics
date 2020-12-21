"""
Liquid argon:
_____________
mass = 6.69E-26 kilograms
distance = sigma = 0.341E-9 meters
energy = epsilon = 1.65E-21 joules
temperature = epsilon / k_B = 119.8 kelvin
tau = sigma * sqrt(mass / epsilon) = 2.17E-12 seconds
velocity = sigma / tau = 1.57E2 m/s
force = epsilon/sigma = 4.85E-12 newtons
pressure = epsilon / (sigma ^ 3) = 4.2E7 pascal = 4.2E2 atmospheres

"""


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
from scripts.helpers import get_empty_float_scalars
from scripts.log_config import debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters
from scripts.numba_procedures import get_interparticle_distances
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
        self.verlet = Verlet(
            static=self.static,
            dynamic=self.dynamic,
            external=external,
            model=self.model,
            potential=self.potential,
        )
        self.saver = Saver(
            static=self.static,
            dynamic=self.dynamic,
            model=self.model,
            **config_parameters['saver_parameters'],
        )
        self.rdf_parameters = config_parameters['rdf_parameters']
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
        self.dynamic.old_positions = deepcopy(self.dynamic.positions)
        system_kinetic_energy, temperature = self.verlet.system_dynamics(
            stage_id=1,
            environment_type=self.environment_type,
        )
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
        parameters['system_kinetic_energy'] = self.dynamic.system_kinetic_energy
        parameters['temperature'] = self.dynamic.temperature(
            system_kinetic_energy=parameters['system_kinetic_energy']
        )
        parameters['pressure'] = pressure
        parameters['total_energy'] = total_energy
        debug_info(f'Kinetic Energy after system_dynamics_2: {self.dynamic.system_kinetic_energy}')
        debug_info(f'Temperature after system_dynamics_2: {parameters["temperature"]}')
        debug_info(f'Pressure after system_dynamics_2: {pressure}')
        debug_info(f'Potential energy after system_dynamics_2: {potential_energy}')
        debug_info(f'Total energy after system_dynamics_2: {total_energy}')
        debug_info(f'Virial after system_dynamics_2: {virial}')
        if not is_rdf_calculation and system_parameters is not None:
            msd = self.dynamic.get_msd(
                previous_positions=self.dynamic.first_positions,
            )
            diffusion = msd / 6 / self.model.time_step / step
            parameters['msd'] = msd
            parameters['diffusion'] = diffusion
            debug_info(f'MSD after system_dynamics_2: {msd}')
            debug_info(f'Diffusion after system_dynamics_2: {diffusion}')

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

    def reduce_transition_processes(self):
        print('Reducing Transition Processes.')
        debug_info('Reducing Transition Processes.')
        external_temperature = self.verlet.external.temperature
        self.verlet.external.temperature = round(self.dynamic.temperature(), 5)
        if self.verlet.external.temperature == 0:
            self.verlet.external.temperature = self.model.initial_temperature
        for _ in range(50):
            self.md_time_step(
                potential_table=self.potential.potential_table,
                step=1,
                is_rdf_calculation=True,
            )
        self.verlet.external.temperature = external_temperature

    @logger_wraps()
    def run_md(self):
        start = datetime.now()
        system_parameters = {
            'temperature': get_empty_float_scalars(self.model.iterations_numbers),
            'pressure': get_empty_float_scalars(self.model.iterations_numbers),
            'system_kinetic_energy': get_empty_float_scalars(self.model.iterations_numbers),
            'potential_energy': get_empty_float_scalars(self.model.iterations_numbers),
            'total_energy': get_empty_float_scalars(self.model.iterations_numbers),
            'virial': get_empty_float_scalars(self.model.iterations_numbers),
            'msd': get_empty_float_scalars(self.model.iterations_numbers),
            'diffusion': get_empty_float_scalars(self.model.iterations_numbers),
        }

        self.reduce_transition_processes()
        self.dynamic.first_positions = deepcopy(self.dynamic.positions)

        for step in range(1, self.model.iterations_numbers + 1):
            self.model.time += self.model.time_step
            debug_info(f'Step: {step}; Time: {self.model.time:.3f};')
            self.md_time_step(
                potential_table=self.potential.potential_table,
                step=step,
                system_parameters=system_parameters,
            )
            print(
                f'Step: {step}/{self.model.iterations_numbers};',
                f'\tTime = {self.model.time:.3f};',
                f'\tT = {system_parameters["temperature"][step - 1]:.5f};',
                f'\tP = {system_parameters["pressure"][step - 1]:.5f};\n',
                sep='\n',
            )
            debug_info(f'End of step {step}.\n')

            if self.is_rdf_calculated:
                if step in (1, 1000) or step % self.rdf_parameters['rdf_saving_step'] == 0:
                    self.run_rdf(virial=system_parameters['virial'][step - 1])

        self.saver.save_configurations(
            is_last_step=True,
        )
        print(f'Calculation completed. Time of calculation: {datetime.now() - start}')
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
        )
        self.saver.save_configuration()

    def run_rdf(
            self,
            virial: float = None,
            layer_thickness: float = 0.01,
    ):
        start = time()
        begin_step = self.rdf_parameters['equilibration_steps'] + 1
        end_step = begin_step - 1 + self.rdf_parameters['calculation_steps']
        sample = deepcopy(self)
        rdf = get_empty_float_scalars(20 * sample.static.particles_number)
        print(f'********RDF calculation for T = {sample.dynamic.temperature():.5f}********')
        isotherm_system_parameters = {
            'msd': get_empty_float_scalars(end_step),
            'diffusion': get_empty_float_scalars(end_step),
        }
        sample.verlet.external.temperature = round(self.dynamic.temperature(), 5)
        if sample.verlet.external.temperature == 0:
            sample.verlet.external.temperature = sample.model.initial_temperature
        sample.verlet.external.pressure = self.dynamic.get_pressure(
            virial=virial,
            temperature=sample.verlet.external.temperature,
        )
        debug_info(f'External Temperature: {sample.verlet.external.temperature}')
        debug_info(f'External Pressure: {sample.verlet.external.pressure}')
        sample.dynamic.first_positions = deepcopy(sample.dynamic.positions)
        for rdf_step in range(1, end_step + 1):
            message = (
                f'RDF Step: {rdf_step}/{end_step}, '
                f'Temperature = {sample.dynamic.temperature():.5f} epsilon/k_B'
            )
            if rdf_step == begin_step:
                print(f'********RDF Calculation started********')
            debug_info(message)
            print(message)
            if rdf_step >= begin_step:
                distances = get_interparticle_distances(
                    distances=np.zeros(
                        (sample.static.particles_number, sample.static.particles_number),
                        dtype=np.float,
                    ),
                    positions=sample.dynamic.positions,
                    cell_dimensions=sample.static.cell_dimensions,
                )
                bins = np.arange(layer_thickness, distances.max() + 1, layer_thickness)
                hist = np.histogram(
                    distances.flatten(),
                    bins
                )[0]
                rdf[:hist.size] += (
                        2.0 * sample.static.get_cell_volume()
                        / (4.0 * pi * bins[:hist.size] ** 2
                           * sample.static.particles_number * sample.static.particles_number)
                        * hist / layer_thickness
                )

            sample.md_time_step(
                potential_table=sample.potential.potential_table,
                step=rdf_step,
                is_rdf_calculation=True,
            )
            msd = sample.dynamic.get_msd(
                previous_positions=sample.dynamic.first_positions,
            )
            diffusion = msd / 6 / sample.model.time_step / rdf_step
            isotherm_parameters = {
                'msd': msd,
                'diffusion': diffusion,
            }
            debug_info(f'MSD: {msd}')
            debug_info(f'Diffusion: {diffusion}')
            sample.saver.step = rdf_step
            sample.saver.update_system_parameters(
                system_parameters=isotherm_system_parameters,
                parameters=isotherm_parameters,
            )

        rdf = rdf[:np.nonzero(rdf)[0][-1]] / (end_step - begin_step + 1)
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


def main(
        config_filename: str = None,
        is_initially_frozen: bool = True,
        is_rdf_calculated: bool = True,
):
    MolecularDynamics(
        config_filename=config_filename,
        is_initially_frozen=is_initially_frozen,
        is_rdf_calculated=is_rdf_calculated
    ).run_md()


if __name__ == '__main__':
    np.set_printoptions(threshold=5000)
    # md_sample = MolecularDynamics(
    #     is_initially_frozen=True,
    # )
    # distances = np.zeros(
    #     (md_sample.static.particles_number, md_sample.static.particles_number),
    #     dtype=np.float,
    # )
    #
    #
    # def f(x):
    #     for _ in range(x):
    #         get_interparticle_distances(
    #             distances=distances,
    #             positions=md_sample.dynamic.positions,
    #             cell_dimensions=md_sample.static.cell_dimensions,
    #         )

    # f(1)
    # run('f(1)')

    # run(
    #     'md_sample.main()',
    #     sort=2,
    # )
    main(
        # TODO check potential at T = 2.8 (compare 2020-12-17 and the book, p.87)
        # config_filename='book_chapter_4_stage_1.json',
        # config_filename='book_chapter_4_stage_2.json',
        # config_filename='cooling.json',
        config_filename='equilibrium_2.8.json',
        # config_filename='equilibrium_0.01.json',
        is_initially_frozen=False,
        is_rdf_calculated=True,
    )
