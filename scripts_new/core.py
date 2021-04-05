from copy import deepcopy
from datetime import datetime
from json import load
from os.path import join
from typing import Optional

import numpy as np

from scripts_new.integrators.npt_mttk import MTTK
from scripts_new.integrators.nve import NVE
from scripts_new.integrators.nvt_velocity_scaling import VelocityScaling
from scripts_new.accelerations_calculator import AccelerationsCalculator
from scripts_new.constants import PATH_TO_CONFIG
from scripts_new.external_parameters import ExternalParameters
from scripts_new.immutable_parameters import ImmutableParameters
from scripts_new.isotherm import Isotherm
from scripts_new.initializer import Initializer
from scripts_new.numba_procedures import get_radius_vectors
from scripts_new.saver import Saver
from scripts_new.simulation_parameters import SimulationParameters
from scripts_new.system import System
from scripts_new.helpers import math_round, get_parameters_dict, print_info
from scripts_new.log_config import log_debug_info, logger_wraps


class MolecularDynamics:

    def __init__(
            self,
            config_filename: Optional[str] = None,
            is_with_isotherms: bool = True,
            is_msd_calculated: bool = True,
    ):
        _config_filename = join(
            PATH_TO_CONFIG,
            config_filename or 'config.json'
        )
        with open(_config_filename, encoding='utf8') as file:
            config_parameters = load(file)

        self.system = System()
        self.initials = Initializer(
            system=self.system,
            **config_parameters["initials"],
        ).get_initials()
        self.immutables = ImmutableParameters(
            particles_number=self.initials.configuration.particles_number,
            **config_parameters["immutables"],
        )
        self.externals = ExternalParameters(**config_parameters["externals"])
        self.integrator = self.get_integrator()
        self.accelerations_calculator = AccelerationsCalculator(
            system=self.initials,
            immutables=self.immutables,
        )
        self.sim_parameters = SimulationParameters(
            **config_parameters['simulation_parameters']
        )
        self.saver = Saver(
            system=self.system,
            simulation_parameters=self.sim_parameters,
        )
        self.is_with_isotherms = is_with_isotherms
        self.is_msd_calculated = is_msd_calculated
        self.interparticle_vectors = np.zeros(
            (
                self.system.configuration.particles_number,
                self.system.configuration.particles_number,
                3
            ),
            dtype=np.float32,
        )
        self.interparticle_distances = np.zeros(
            (
                self.system.configuration.particles_number,
                self.system.configuration.particles_number,
            ),
            dtype=np.float32,
        )

    def calculate_interparticle_vectors(self):
        self.interparticle_vectors, self.interparticle_distances = get_radius_vectors(
            radius_vectors=self.interparticle_vectors,
            positions=self.system.configuration.positions,
            cell_dimensions=self.system.cell_dimensions,
            distances=self.interparticle_distances,
        )

    def get_integrator(self):
        integrator = {
            'mttk': MTTK,
            'velocity_scaling': VelocityScaling,
        }.get(self.externals.environment_type, NVE)
        return integrator(
            system=self.initials,
            time_step=self.immutables.time_step,
            external=self.externals,
        )

    def md_time_step(
            self,
            step: int,
            system_parameters: dict = None,
            is_rdf_calculation: bool = False,
            is_pbc_switched_on: bool = False,
    ):
        parameters = {}
        self.integrator.system = self.system
        self.integrator.stage_1()
        kinetic_energy, _ = self.integrator.after_stage(1)
        if is_pbc_switched_on:
            self.system.apply_boundary_conditions()
        self.accelerations_calculator.system = self.system
        self.accelerations_calculator.load_forces()
        parameters['kinetic_energy'] = kinetic_energy
        parameters['potential_energy'] = self.system.potential_energy
        self.integrator.system = self.system
        self.integrator.stage_2()
        _, _, pressure, total_energy = self.integrator.after_stage(
            stage_id=2
        )
        parameters.update({
            'kinetic_energy': self.system.configuration.kinetic_energy,
            'temperature': self.system.configuration.get_temperature(
                kinetic_energy=parameters['kinetic_energy']
            ),
            'pressure': pressure,
            'total_energy': total_energy,
            'virial': self.system.virial,
        })
        if not is_rdf_calculation and system_parameters is not None:
            if self.is_msd_calculated:
                msd = self.system.configuration.get_msd(
                    previous_positions=self.initials.configuration.positions,
                )
                diffusion = msd / 6 / self.immutables.time_step / step
                parameters['msd'] = msd
                parameters['diffusion'] = diffusion
                log_debug_info(f'MSD after system_dynamics_2: {msd}')
                log_debug_info(
                    f'Diffusion after system_dynamics_2: {diffusion}'
                )

            self.saver.system = self.system
            self.saver.step = step
            self.saver.update_system_parameters(
                system_parameters=system_parameters,
                parameters=parameters,
            )
            self.saver.store_configuration()
            self.saver.save_configurations()

    def fix_current_temperature(self):
        self.externals.temperature = math_round(
            number=self.system.configuration.get_temperature(),
            number_of_digits_after_separator=5,
        )
        if self.externals.temperature == 0.0:
            _initial_temperature = self.initials.configuration.temperature
            self.externals.temperature = _initial_temperature

    def reduce_transition_processes(
            self,
            skipped_iterations: int = 50,
    ):
        print('Reducing Transition Processes.')
        log_debug_info('Reducing Transition Processes.')
        external_temperature = self.externals.temperature
        self.fix_current_temperature()
        for _ in range(skipped_iterations):
            self.md_time_step(
                step=1,
                is_rdf_calculation=True,
            )
        self.externals.temperature = external_temperature

    def fix_external_conditions(self):
        print(
            f'*******Isotherm for '
            f'T = {self.system.configuration.get_temperature():.5f}'
            f'*******'
        )
        self.fix_current_temperature()
        self.externals.pressure = self.system.get_pressure(
            temperature=self.externals.temperature,
        )
        log_debug_info(f'External Temperature: {self.externals.temperature}')
        log_debug_info(f'External Pressure: {self.externals.pressure}')

    def equilibrate_system(self, equilibration_steps: int):
        for eq_step in range(equilibration_steps):
            temperature = self.system.configuration.get_temperature()
            pressure = self.system.get_pressure(temperature=temperature)
            message = (
                f'Equilibration Step: {eq_step:3d}/{equilibration_steps}, \t'
                f'Temperature = {temperature:8.5f} epsilon/k_B, \t'
                f'Pressure = {pressure:.5f} epsilon/sigma^3, \t'
            )
            log_debug_info(message)
            print(message)
            self.md_time_step(
                step=1,
                is_rdf_calculation=True,
            )

    def save_all(self, system_parameters):
        self.saver.save_configurations(
            is_last_step=True,
        )
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
        )
        self.saver.save_configuration()

    @logger_wraps()
    def run_md(self):
        start = datetime.now()
        system_parameters = get_parameters_dict(
            names=(
                'temperature',
                'pressure',
                'kinetic_energy',
                'potential_energy',
                'total_energy',
                'virial',
                'msd',
                'diffusion',
            ),
            value_size=self.sim_parameters.iterations_numbers,
        )

        # self.reduce_transition_processes()
        # self.dynamic.first_positions = deepcopy(self.dynamic.positions)

        for step in range(1, self.sim_parameters.iterations_numbers + 1):
            self.system.time += self.immutables.time_step
            log_debug_info(f'Step: {step}; Time: {self.system.time:.3f};')
            self.md_time_step(
                step=step,
                system_parameters=system_parameters,
                is_pbc_switched_on=True,
            )
            print_info(
                step=step,
                iterations_numbers=self.sim_parameters.iterations_numbers,
                current_time=self.system.time,
                parameters=system_parameters,
            )
            log_debug_info(f'End of step {step}.\n')

            if self.is_with_isotherms:
                isotherm_steps = (
                    1,
                    # 1000,
                )
                if (
                        (
                                step % self.sim_parameters.isotherm_saving_step
                        ) == 0 or step in isotherm_steps
                ):
                    Isotherm(
                        sample=deepcopy(self),
                    ).run()
        self.save_all(
            system_parameters=system_parameters,
        )
        print(
            f'Simulation is completed. '
            f'Time of calculation: {datetime.now() - start}'
        )


if __name__ == '__main__':
    MD = MolecularDynamics('test.json')
