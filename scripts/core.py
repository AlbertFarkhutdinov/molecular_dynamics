from copy import deepcopy
from datetime import datetime
from typing import Optional

import numpy as np

from scripts.integrators.npt_mttk import MTTK
from scripts.integrators.nve import NVE
from scripts.integrators.nvt_velocity_scaling import VelocityScaling
from scripts.accelerations_calculator import AccelerationsCalculator
from scripts.external_parameters import ExternalParameters
from scripts.immutable_parameters import ImmutableParameters
from scripts.isotherm import Isotherm
from scripts.initializer import Initializer
from scripts.helpers import get_config_parameters
from scripts.numba_procedures import get_radius_vectors
from scripts.saver import Saver
from scripts.simulation_parameters import SimulationParameters
from scripts.system import System
from scripts.helpers import math_round, get_parameters_dict, print_info
from scripts.log_config import log_debug_info, logger_wraps


class MolecularDynamics:

    def __init__(
            self,
            config_filename: Optional[str],
            is_with_isotherms: bool = True,
            is_msd_calculated: bool = True,
    ):
        print(f'Initialization of {self.__class__.__name__} instance.')
        config_parameters = get_config_parameters(config_filename)

        self.system = System()
        print(f'System is created.')
        self.initials = Initializer(
            system=self.system,
            **config_parameters["initials"],
        ).get_initials()
        print(f'Initial parameters are received.')
        self.immutables = ImmutableParameters(
            particles_number=self.initials.configuration.particles_number,
            **config_parameters["immutables"],
        )
        print(f'Immutable parameters are received.')
        self.accelerations_calculator = AccelerationsCalculator(
            system=self.initials,
            immutables=self.immutables,
        )
        print(f'Accelerations Calculator is initialized.')
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
        self.externals, self.integrator = None, None
        self.sim_parameters, self.saver = None, None
        self.update_simulation_parameters(config_parameters)
        print(f'Simulation parameters are updated.')

    def update_simulation_parameters(self, config_parameters):
        self.externals = ExternalParameters(**config_parameters["externals"])
        self.integrator = self.get_integrator()
        self.sim_parameters = SimulationParameters(
            **config_parameters['simulation_parameters']
        )
        self.saver = Saver(
            system=self.system,
            simulation_parameters=self.sim_parameters,
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
            'volume': self.system.volume,
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
        time = str(datetime.now()).split('.')[0].replace(
            ' ', '_'
        ).replace(
            ':', '_'
        ).replace(
            '-', '_'
        )
        self.saver.save_configurations(
            is_last_step=True,
        )
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
            file_name=f'system_parameters_{time}.csv',
        )
        self.saver.save_configuration(
            file_name=f'system_configuration_{time}.csv',
        )

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
                'volume',
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
                isotherm_steps = (1, )
                if (
                        not (step % self.sim_parameters.isotherm_saving_step)
                        or step in isotherm_steps
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
    MD = MolecularDynamics('nve.json')
