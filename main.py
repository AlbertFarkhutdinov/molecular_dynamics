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


from cProfile import run
from json import load
from math import pi
from datetime import datetime
from os import getcwd
from os.path import join
from typing import Optional

import numpy as np

from constants import PATH_TO_DATA
from dynamic_parameters import SystemDynamicParameters
from external_parameters import ExternalParameters
from helpers import get_empty_float_scalars, get_empty_int_scalars
from log_config import debug_info, logger_wraps
from modeling_parameters import ModelingParameters
from potential_parameters import PotentialParameters
from saver import Saver
from static_parameters import SystemStaticParameters
from verlet import Verlet


class MolecularDynamics:

    def __init__(
            self,
            config_filename: Optional[str] = None,
            is_initially_frozen: bool = True,
    ):
        _config_filename = join(
            getcwd(),
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

    def md_time_step(
            self,
            potential_table: np.ndarray,
            step: int,
            system_parameters: dict,
    ):
        system_kinetic_energy, temperature = self.verlet.system_dynamics(
            stage_id=1,
            thermostat_type='velocity_scaling',
        )
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
        self.saver.dynamic = self.dynamic
        self.saver.step = step
        self.saver.model.time = self.model.time
        self.saver.update_system_parameters(
            system_parameters=system_parameters,
            temperature=temperature,
            **parameters
        )
        self.saver.store_configuration()
        print(
            f'Step: {step}/{self.model.iterations_numbers};',
            f'\tTime = {self.model.time:.3f};',
            f'\tT = {temperature:.5f};',
            f'\tP = {pressure:.5f};\n',
            sep='\n',
        )
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
            self.model.time += self.model.time_step
            debug_info(f'Step: {step}; Time: {self.model.time:.3f};')
            self.md_time_step(
                potential_table=self.potential.potential_table,
                step=step,
                system_parameters=system_parameters,
            )
            debug_info(f'End of step {step}.\n')

        self.saver.save_configurations(
            is_last_step=True,
        )
        print(f'Calculation completed. Time of calculation: {datetime.now() - start}')
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
        )

    def run_rdf(self, steps_number: int = 1000, file_name: str = None):
        file_name = join(PATH_TO_DATA, file_name or 'system_config.txt')
        layer_thickness = 0.01
        begin_step = 1
        end_step = steps_number
        rdf = get_empty_float_scalars(10 * self.static.particles_number)
        with open(file_name, mode='r', encoding='utf8') as file:
            lines = file.readlines()
            for step in range(begin_step, end_step + 1):
                for i in range(self.static.particles_number):
                    self.dynamic.positions[i] = np.array(
                        lines[9 + i].split()[2:],
                        dtype=np.float,
                    )
                distances = self.dynamic.interparticle_distances
                layer_numbers = (distances / layer_thickness).astype(np.int)
                max_layer_number = layer_numbers.max()
                particles_in_layer = get_empty_int_scalars(max_layer_number + 1)
                layers, particles = np.unique(layer_numbers, return_counts=True)
                for i, layer in enumerate(layers):
                    particles_in_layer[layer] = particles[i]
                radiuses = layer_thickness * (np.arange(max_layer_number + 1) + 0.5)
                rdf[:max_layer_number + 1] += (
                        2.0 * self.static.get_cell_volume()
                        / (4.0 * pi * radiuses * radiuses
                           * self.static.particles_number * self.static.particles_number)
                        * particles_in_layer / layer_thickness
                )
                print(f'Step: {step}/{end_step};')
        with open('rdf_file.txt', mode='w', encoding='utf8') as file:
            rdf = rdf[:np.nonzero(rdf)[0][-1]]
            radiuses = layer_thickness * (np.arange(rdf.size) + 0.5)
            rdf = rdf / (end_step - begin_step + 1)
            for i, radius in enumerate(radiuses):
                if radius <= self.static.cell_dimensions[0] / 2.0:
                    file.write(f'{radius} {rdf[i]}\n')
        print('Calculation completed.')


def main(
        config_filename: str = None,
        is_initially_frozen: bool = True,
        is_profiled: bool = False,
):
    md_sample = MolecularDynamics(
        config_filename=config_filename,
        is_initially_frozen=is_initially_frozen,
    )
    if is_profiled:
        run(
            'md_sample.run_md()',
            sort=2,
        )
    else:
        md_sample.run_md()


if __name__ == '__main__':
    np.set_printoptions(threshold=5000)
    main()
