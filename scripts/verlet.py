import numpy as np

from scripts.constants import TEMPERATURE_MINIMUM
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import get_empty_float_scalars, get_empty_int_scalars, get_empty_vectors, sign
from scripts.log_config import debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters
from scripts.numba_procedures import lf_cycle, update_list_cycle
from scripts.potential_parameters import PotentialParameters
from scripts.static_parameters import SystemStaticParameters


class Verlet:

    def __init__(
            self,
            static: SystemStaticParameters,
            dynamic: SystemDynamicParameters,
            external: ExternalParameters,
            model: ModelingParameters,
            potential: PotentialParameters,
    ):
        self.potential = potential
        self.static = static
        self.dynamic = dynamic
        self.external = external
        self.model = model
        self.neighbours_lists = {
            'all_neighbours': get_empty_int_scalars(self.static.particles_number),
            'first_neighbours': get_empty_int_scalars(self.static.particles_number),
            'last_neighbours': get_empty_int_scalars(100 * self.static.particles_number),
        }
        self.nvt_factor = 0
        self.npt_factor = 0
        self.s_f = 0

    @logger_wraps()
    def system_dynamics(
            self,
            stage_id: int,
            environment_type: str,
            **kwargs,
    ):
        if stage_id == 1:
            return self.stage_1(algorithm_name=environment_type)
        if stage_id == 2:
            if environment_type == 'velocity_scaling':
                return self.velocity_scaling_2(**kwargs)
            elif environment_type == 'nose_hoover':
                return self.nose_hoover_2(**kwargs)

    @logger_wraps()
    def stage_1(self, algorithm_name: str):
        acc_coefficient = 0
        if algorithm_name == 'nose_hoover':
            acc_coefficient = self.nvt_factor + self.npt_factor

        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
            acc_coefficient=acc_coefficient,
        )
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        self.model.initial_temperature += (
                sign(self.external.temperature - self.model.initial_temperature)
                * self.external.heating_velocity
                * self.model.time_step
        )
        if self.model.initial_temperature < TEMPERATURE_MINIMUM:
            self.model.initial_temperature = TEMPERATURE_MINIMUM
        if temperature <= 0:
            temperature = self.model.initial_temperature

        if algorithm_name == 'velocity_scaling':
            self.nvt_factor = (self.model.initial_temperature / temperature) ** 0.5
        elif algorithm_name == 'berendsen':
            self.nvt_factor = (1 + self.model.time_step / self.external.thermostat_parameter
                               * (self.model.initial_temperature / temperature - 1)) ** 0.5
        elif algorithm_name == 'nose_hoover':
            lambda_diff = ((2.0 * system_kinetic_energy
                            - 3.0 * self.static.particles_number * self.model.initial_temperature)
                           * self.model.time_step / self.external.thermostat_parameter)
            self.s_f += self.nvt_factor * self.model.time_step + lambda_diff * self.model.time_step
            self.nvt_factor += lambda_diff / 2.0

        debug_info(f'Kinetic Energy: {system_kinetic_energy};')
        debug_info(f'Temperature before {algorithm_name}_1: {temperature};')
        debug_info(f'Initial Temperature: {self.model.initial_temperature};')
        debug_info(f'NVT-factor: {self.nvt_factor};')

        vel_coefficient = 1
        if algorithm_name == 'velocity_scaling':
            vel_coefficient = self.nvt_factor
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            vel_coefficient=vel_coefficient,
        )
        debug_info(f'Temperature after {algorithm_name}_1: {self.dynamic.temperature()};')
        return system_kinetic_energy, temperature

    @logger_wraps()
    def velocity_scaling_2(
            self,
            virial: float,
            temperature: float,
            potential_energy: float,
            system_kinetic_energy: float,
    ):
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            virial=virial,
        )
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        debug_info(f'Temperature before velocity_scaling_2: {temperature};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
        total_energy = system_kinetic_energy + potential_energy
        debug_info(f'Temperature after velocity_scaling_2: {self.dynamic.temperature()};')
        return pressure, total_energy

    @logger_wraps()
    def nose_hoover_2(
            self,
            virial: float,
            temperature: float,
            potential_energy: float,
            system_kinetic_energy: float,
    ):
        cell_volume = self.static.get_cell_volume()
        density = self.static.get_density()
        pressure = self.dynamic.get_pressure(
            virial=virial,
            temperature=temperature,
            cell_volume=cell_volume,
        )
        debug_info(f'Temperature before nose_hoover_2: {temperature};')
        debug_info(f'Pressure before nose_hoover_2: {pressure};')
        # TODO incorrect algorithm
        self.npt_factor += (
                (pressure - self.external.pressure)
                * self.model.time_step
                / self.external.barostat_parameter / density
        )

        self.static.cell_dimensions *= 1.0 + self.npt_factor * self.model.time_step

        # cell_volume = self.static.get_cell_volume()
        # density = self.static.get_density()

        error = 1e-5
        half_time = self.model.time_step / 2.0
        time_ratio = self.model.time_step / self.external.thermostat_parameter
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        nvt_factor_new = self.nvt_factor
        velocities_new = self.dynamic.velocities
        is_ready = False
        for i in range(50):
            if is_ready:
                break
            nvt_factor_old = nvt_factor_new
            ds = 0
            velocities_old = velocities_new
            b = (
                    -half_time
                    * (self.dynamic.accelerations - nvt_factor_old * self.dynamic.velocities)
                    - (self.dynamic.velocities - velocities_old)
            )
            ds += (velocities_old * b).sum() * time_ratio
            dd = 1 - nvt_factor_old * half_time
            ds -= dd * (
                    (3.0 * self.static.particles_number * self.model.initial_temperature - 2.0 * system_kinetic_energy)
                    * 2.0 * time_ratio
                    - (self.nvt_factor - nvt_factor_old)
            )
            ds = ds / (-self.model.time_step * system_kinetic_energy * time_ratio + dd)
            velocities_new = self.dynamic.velocities + (b + half_time * velocities_old * ds) / dd
            system_kinetic_energy = (velocities_new * velocities_new).sum() / 2.0
            nvt_factor_new = nvt_factor_old + ds
            is_ready = not (
                    (abs((velocities_new - velocities_old) / velocities_new) > error).any()
                    or (abs((nvt_factor_new - nvt_factor_old) / nvt_factor_new) > error).any()
            )

        self.dynamic.velocities = velocities_new
        self.nvt_factor = nvt_factor_new
        total_energy = (
                system_kinetic_energy + potential_energy
                + self.nvt_factor * self.nvt_factor * self.external.thermostat_parameter / 2.0
                + 3.0 * self.static.particles_number * self.model.initial_temperature * self.s_f
        )

        cell_volume = self.static.get_cell_volume()
        temperature = self.dynamic.temperature()
        pressure = self.dynamic.get_pressure(
            virial=virial,
            temperature=temperature,
            cell_volume=cell_volume,
        )
        debug_info(f'Temperature after nose_hoover_2: {temperature};')
        debug_info(f'Pressure after nose_hoover_2: {pressure};')
        return pressure, total_energy

    def load_forces(
            self,
            potential_table: np.ndarray,
    ):
        debug_info(f"Entering 'load_forces(potential_table)'")
        potential_energies = get_empty_float_scalars(self.static.particles_number)
        self.dynamic.accelerations = get_empty_vectors(self.static.particles_number)
        if self.potential.update_test:
            debug_info(f'update_test = True')
            self.update_list()
            self.dynamic.displacements = get_empty_vectors(self.static.particles_number)
            self.potential.update_test = False

        virial = lf_cycle(
            particles_number=self.static.particles_number,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
            r_cut=self.potential.r_cut,
            potential_table=potential_table,
            potential_energies=potential_energies,
            positions=self.dynamic.positions,
            accelerations=self.dynamic.accelerations,
            cell_dimensions=self.static.cell_dimensions,
        )
        acc_mag = (self.dynamic.accelerations ** 2).sum(axis=1) ** 0.5
        debug_info(f'Mean and max acceleration: {acc_mag.mean()}, {acc_mag.max()}')
        # TODO Check potential calculation (compare 2020-11-21 and the book, p.87)
        potential_energy = potential_energies.sum()
        self.dynamic.displacements += (
                self.dynamic.velocities * self.model.time_step
                + self.dynamic.accelerations * self.model.time_step * self.model.time_step / 2.0
        )
        self.load_move_test()
        debug_info(f'Potential energy: {potential_energy};')
        debug_info(f"Exiting 'load_forces(potential_table)'")
        return potential_energy, virial

    @logger_wraps()
    def update_list(self):
        self.neighbours_lists = {
            'first_neighbours': get_empty_int_scalars(self.static.particles_number),
            'last_neighbours': get_empty_int_scalars(self.static.particles_number),
            'all_neighbours': get_empty_int_scalars(100 * self.static.particles_number),
        }
        advances = get_empty_int_scalars(self.static.particles_number)
        update_list_cycle(
            rng=self.potential.r_cut + self.potential.skin,
            advances=advances,
            particles_number=self.static.particles_number,
            positions=self.dynamic.positions,
            cell_dimensions=self.static.cell_dimensions,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
        )

    @logger_wraps()
    def load_move_test(self):
        ds_1, ds_2 = 0.0, 0.0
        for i in range(self.static.particles_number):
            ds = (self.dynamic.displacements[i] ** 2).sum() ** 0.5
            if ds >= ds_1:
                ds_2 = ds_1
                ds_1 = ds
            elif ds >= ds_2:
                ds_2 = ds
        self.potential.update_test = (ds_1 + ds_2) > self.potential.skin
