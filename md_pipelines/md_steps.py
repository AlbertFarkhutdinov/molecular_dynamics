from copy import deepcopy

from math_round_af import get_rounded_number
import numpy as np

from common.helpers import get_json, get_current_time, save_config_parameters
from common.helpers import get_parameters_dict, print_info
from common.numba_procedures import get_radius_vectors
from configurations import ThermodynamicSystem
from core.accelerations_calculator import AccelerationsCalculator
from core.external_parameters import ExternalParameters
from core.immutable_parameters import ImmutableParameters
from core.isotherm import Isotherm
from core.saver import Saver
from core.simulation_parameters import SimulationParameters
from integrators import IsobarMTTK, IsothermMTTK, NVE, VelocityScaling
from logs import LoggedObject


class MolecularDynamicsSteps(LoggedObject):

    def __init__(
            self,
            config_filenames: dict[str, str],
            is_with_isotherms: bool = True,
            is_msd_calculated: bool = True,
    ):
        self.logger.info(
            f'{self.__class__.__name__} instance initialization.'
        )
        self.config_parameters = {
            key: get_json(key, value)
            for key, value in config_filenames.items()
        }
        initializer = Initializer(
            **self.config_parameters["initials"],
        ).get_initials()
        self.system = initializer.initialize()
        self.initial_system = ThermodynamicSystem(
            positions=self.system.positions,
            cell_dimensions=self.system.cell_dimensions,
            is_pbc_applied=self.system.is_pbc_applied,
            time=self.system.time,
            velocities=self.system.velocities,
            accelerations=self.system.accelerations,
        )
        self.logger.info('System is created.')
        self.logger.info('Initial parameters are received.')
        self.immutables = ImmutableParameters(
            particles_number=self.initial_system.particles_number,
            **self.config_parameters["immutables"],
        )
        self.logger.info('Immutable parameters are received.')
        self.accelerations_calculator = AccelerationsCalculator(
            system=self.initial_system,
            immutables=self.immutables,
        )
        self.logger.info('Accelerations Calculator is initialized.')
        self.is_with_isotherms = is_with_isotherms
        self.is_msd_calculated = is_msd_calculated
        self.interparticle_vectors = np.zeros(
            (
                self.system.particles_number,
                self.system.particles_number,
                3
            ),
            dtype=np.float32,
        )
        self.interparticle_distances = np.zeros(
            (
                self.system.particles_number,
                self.system.particles_number,
            ),
            dtype=np.float32,
        )
        self.externals, self.integrator = None, None
        self.sim_parameters, self.saver = None, None
        self.update_simulation_parameters(self.config_parameters)
        self.logger.info('Simulation parameters are updated.')

    def update_simulation_parameters(self, config_parameters):
        self.externals = ExternalParameters(
            **config_parameters.get("externals", dict())
        )
        self.integrator = self.get_integrator()
        self.sim_parameters = SimulationParameters(
            **config_parameters['simulation_parameters']
        )
        self.saver = Saver(
            system=self.system,
            simulation_parameters=self.sim_parameters,
            parameters_saving_step=100,
        )

    def calculate_interparticle_vectors(self):
        self.interparticle_vectors, self.interparticle_distances = get_radius_vectors(
            radius_vectors=self.interparticle_vectors,
            positions=self.system.positions,
            cell_dimensions=self.system.cell_dimensions,
            distances=self.interparticle_distances,
        )

    def get_integrator(self):
        integrator = {
            'mtk_npt': IsobarMTTK,
            'mttk_npt': IsobarMTTK,
            'mtk_nvt': IsothermMTTK,
            'mttk_nvt': IsothermMTTK,
            'velocity_scaling': VelocityScaling,
            'velocity_rescaling': VelocityScaling,
        }.get(self.externals.environment_type, NVE)
        return integrator(
            system=self.initial_system,
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
            'time': self.system.time,
            'kinetic_energy': self.system.kinetic_energy,
            'temperature': self.system.temperature,
            'pressure': pressure,
            'total_energy': total_energy,
            'vir': self.system.vir,
            'volume': self.system.volume,
        })
        if not is_rdf_calculation and system_parameters is not None:
            if self.is_msd_calculated:
                msd = self.system.configuration.get_msd(
                    previous_positions=self.initial_system.positions,
                )
                diffusion = msd / 6 / self.immutables.time_step / step
                parameters['msd'] = msd
                parameters['diffusion'] = diffusion
                self.logger.debug(f'MSD after system_dynamics_2: {msd}')
                self.logger.debug(
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
        self.externals.temperature = get_rounded_number(
            number=self.system.temperature,
            number_of_digits_after_separator=5,
        )
        if self.externals.temperature == 0.0:
            _initial_temperature = self.initial_system.temperature
            self.externals.temperature = _initial_temperature

    def reduce_transition_processes(
            self,
            skipped_iterations: int = 1000,
    ):
        self.logger.info('Reducing Transition Processes.')
        external_temperature = self.externals.temperature
        self.fix_current_temperature()
        for _ in range(skipped_iterations):
            self.md_time_step(
                step=1,
                is_rdf_calculation=True,
            )
        self.externals.temperature = external_temperature

    def fix_external_conditions(self):
        self.logger.info(
            '*******Isotherm for '
            f'T = {self.system.temperature:.5f}'
            '*******'
        )
        self.fix_current_temperature()
        self.externals.pressure = self.system.get_pressure(
            temperature=self.externals.temperature,
        )
        self.logger.info(f'External Temperature: {self.externals.temperature}')
        self.logger.info(f'External Pressure: {self.externals.pressure}')

    def equilibrate_system(self, equilibration_steps: int):
        for eq_step in range(equilibration_steps):
            temperature = self.system.temperature
            pressure = self.system.pressure
            message = (
                f'Equilibration Step: {eq_step:3d}/{equilibration_steps}, \t'
                f'Temperature = {temperature:8.5f} epsilon/k_B, \t'
                f'Pressure = {pressure:.5f} epsilon/sigma^3, \t'
            )
            self.logger.info(message)
            self.md_time_step(
                step=1,
                is_rdf_calculation=True,
            )

    @staticmethod
    def get_str_time():
        return str(get_current_time()).split('.', maxsplit=1)[0].replace(
            ' ', '_'
        ).replace(
            ':', '_'
        ).replace(
            '-', '_'
        )

    def save_all(self, system_parameters):
        time = self.get_str_time()
        self.saver.save_configurations(is_last_step=True)
        self.saver.save_system_parameters(
            system_parameters=system_parameters,
            file_name=f'system_parameters_{time}.csv',
        )
        self.saver.save_configuration(
            file_name=f'system_configuration_{time}.csv',
        )

    def get_empty_parameters(self):
        return get_parameters_dict(
            names=(
                'time',
                'temperature',
                'pressure',
                'kinetic_energy',
                'potential_energy',
                'total_energy',
                'vir',
                'msd',
                'diffusion',
                'volume',
            ),
            value_size=self.saver.parameters_saving_step,
        )

    def get_rdf_shot(self):
        _rdf = Isotherm(sample=self).rdf
        _rdf.accumulate()
        _rdf.save_sample(sample=_rdf.rdf)

    def run_md(self):
        start = get_current_time()
        system_parameters = self.get_empty_parameters()
        if self.config_parameters["initials"].get("init_type") != -1:
            self.reduce_transition_processes()
        # self.dynamic.first_positions = deepcopy(self.dynamic.positions)

        self.saver.save_configuration(
            file_name=f'system_configuration_{self.get_str_time()}.csv',
        )

        for step in range(1, self.sim_parameters.iterations_numbers + 1):
            self.system.time += self.immutables.time_step
            self.logger.debug(f'Step: {step}; Time: {self.system.time:.3f};')
            self.md_time_step(
                step=step,
                system_parameters=system_parameters,
                is_pbc_switched_on=False,
            )
            print_info(
                step=step,
                step_index=step % self.saver.parameters_saving_step,
                iterations_numbers=self.sim_parameters.iterations_numbers,
                current_time=self.system.time,
                parameters=system_parameters,
            )
            self.logger.debug(f'End of step {step}.\n')
            if step % self.saver.parameters_saving_step == 0:
                self.saver.save_system_parameters(
                    system_parameters=system_parameters,
                    file_name=f'system_parameters_{self.get_str_time()}.csv',
                )
                if step < self.sim_parameters.iterations_numbers:
                    system_parameters = self.get_empty_parameters()
            if self.is_with_isotherms:
                isotherm_steps = (
                    1,
                )
                if (
                        not (step % self.sim_parameters.isotherm_saving_step)
                        or step in isotherm_steps
                ):
                    isotherm = Isotherm(sample=deepcopy(self))
                    isotherm.equilibrate()
                    isotherm.run()

        self.save_all(system_parameters=system_parameters)
        self.logger.info(
            'Simulation is completed. '
            f'Time of calculation: {get_current_time() - start}'
        )
        save_config_parameters(
            config_parameters=self.config_parameters,
            config_number=0,
        )
