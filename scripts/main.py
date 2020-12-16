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
from cProfile import run
from json import load
from math import pi
from datetime import datetime
from os.path import join
from time import time
from typing import Optional

import numpy as np

from scripts.constants import PATH_TO_CONFIG
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import get_empty_float_scalars, get_empty_int_scalars
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
    ):
        _config_filename = join(
            PATH_TO_CONFIG,
            config_filename or 'config.json'
        )
        with open(_config_filename, encoding='utf8') as file:
            config_parameters = load(file)

        self.static = SystemStaticParameters(**config_parameters['static_parameters'])
        self.model = ModelingParameters(**config_parameters['modeling_parameters'])
        self.dynamic = SystemDynamicParameters(
            static=self.static,
            temperature=self.model.initial_temperature if not is_initially_frozen else None,
        )
        self.potential = PotentialParameters(**config_parameters['potential_parameters'])
        self.verlet = Verlet(
            static=self.static,
            dynamic=self.dynamic,
            external=ExternalParameters(**config_parameters['external_parameters']),
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

    def md_time_step(
            self,
            potential_table: np.ndarray,
            step: int,
            system_parameters: dict = None,
            is_rdf_calculation: bool = False,
    ):
        system_kinetic_energy, temperature = self.verlet.system_dynamics(
            stage_id=1,
            thermostat_type='velocity_scaling',
        )
        self.dynamic.boundary_conditions()
        potential_energy, virial = self.verlet.load_forces(
            potential_table=potential_table,
        )
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            virial=virial,
        )
        parameters = {
            'system_kinetic_energy': system_kinetic_energy,
            'potential_energy': potential_energy,
            'pressure': pressure,
        }
        self.verlet.system_dynamics(
            stage_id=2,
            thermostat_type='velocity_scaling',
            **parameters,
        )
        if not is_rdf_calculation and system_parameters is not None:
            self.saver.dynamic = self.dynamic
            self.saver.step = step
            self.saver.model.time = self.model.time
            self.saver.update_system_parameters(
                system_parameters=system_parameters,
                temperature=temperature,
                **parameters
            )
            self.saver.store_configuration()
            self.saver.save_configurations()

    @logger_wraps()
    def run_md(self):
        start = datetime.now()
        system_parameters = {
            'temperature': get_empty_float_scalars(self.model.iterations_numbers),
            'pressure': get_empty_float_scalars(self.model.iterations_numbers),
            'kinetic_energy': get_empty_float_scalars(self.model.iterations_numbers),
            'potential_energy': get_empty_float_scalars(self.model.iterations_numbers),
            'total_energy': get_empty_float_scalars(self.model.iterations_numbers),
        }
        for step in range(1, self.model.iterations_numbers + 1):
            if self.is_rdf_calculated:
                if step in (1, 1000) or step % self.rdf_parameters['rdf_saving_step'] == 0:
                    self.run_rdf()
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

        self.saver.save_configurations(
            is_last_step=True,
        )
        print(f'Calculation completed. Time of calculation: {datetime.now() - start}')
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
        )

    def run_rdf(
            self,
            is_positions_from_file: bool = False
            # file_name: str = None,
    ):
        start = time()
        layer_thickness = 0.01
        begin_step = self.rdf_parameters['equilibration_steps'] + 1
        end_step = begin_step - 1 + self.rdf_parameters['calculation_steps']
        sample = deepcopy(self)
        rdf = get_empty_float_scalars(20 * sample.static.particles_number)
        print(f'********RDF calculation for T = {sample.dynamic.temperature():.5f}********')

        if not is_positions_from_file:
            sample.verlet.external.temperature = round(self.dynamic.temperature(), 5)
            if sample.verlet.external.temperature == 0:
                sample.verlet.external.temperature = sample.model.initial_temperature
            debug_info(f'External Temperature: {sample.verlet.external.temperature}')
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
                    debug_info(f'maximum distance: {distances.max()}')
                    layer_numbers = (distances / layer_thickness).astype(np.int)
                    max_layer_number = layer_numbers.max()
                    debug_info(f'max_layer_number: {max_layer_number}')
                    particles_in_layer = get_empty_int_scalars(max_layer_number + 1)
                    layers, particles = np.unique(layer_numbers, return_counts=True)
                    debug_info(f'layers.shape: {layers.shape}')
                    debug_info(f'particles.shape: {particles.shape}')
                    for i, layer in enumerate(layers):
                        particles_in_layer[layer] = particles[i]
                    debug_info(f'particles_in_layer.shape: {particles_in_layer.shape}')
                    radiuses = layer_thickness * (0.5 + np.arange(max_layer_number + 1))
                    debug_info(f'radiuses.shape: {radiuses.shape}')
                    debug_info(f'rdf.shape: {rdf.shape}')
                    rdf[:max_layer_number + 1] += (
                            2.0 * sample.static.get_cell_volume()
                            / (4.0 * pi * radiuses * radiuses
                               * sample.static.particles_number * sample.static.particles_number)
                            * particles_in_layer / layer_thickness
                    )
                sample.md_time_step(
                    potential_table=sample.potential.potential_table,
                    step=rdf_step,
                    is_rdf_calculation=True,
                )
        else:
            pass
            # TODO implementation of reading from file
            # positions = get_empty_vectors(self.static.particles_number)
            # file_name = join(PATH_TO_DATA, file_name or 'system_config.txt')
            # with open(file_name, mode='r', encoding='utf8') as file:
            #     lines = file.readlines()
            #     for i in range(self.static.particles_number):
            #         positions[i] = np.array(
            #             lines[9 + i].split()[2:],
            #             dtype=np.float,
            #         )

        rdf = rdf[:np.nonzero(rdf)[0][-1]] / (end_step - begin_step + 1)
        debug_info(f'rdf.shape: {rdf.shape}')
        radiuses = layer_thickness * (0.5 + np.arange(rdf.size))
        Saver().save_rdf(
            rdf_data={
                'radius': radiuses[radiuses <= sample.static.cell_dimensions[0] / 2.0],
                'rdf': rdf[radiuses <= sample.static.cell_dimensions[0] / 2.0],
            },
            file_name=f'rdf_file_T_{sample.verlet.external.temperature:.5f}.csv'
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
        # TODO check potential at T = 2.8
        # config_filename='book_chapter_4_stage_1.json',
        config_filename='equilibrium_2.8.json',
        is_initially_frozen=False,
        is_rdf_calculated=False,
    )
