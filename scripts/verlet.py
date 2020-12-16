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
            'marker_1': get_empty_int_scalars(self.static.particles_number),
            'marker_2': get_empty_int_scalars(self.static.particles_number),
            'list': get_empty_int_scalars(100 * self.static.particles_number),
        }

    @logger_wraps()
    def system_dynamics(
            self,
            stage_id: int,
            thermostat_type: str,
            **kwargs,
    ):
        if thermostat_type == 'velocity_scaling':
            if stage_id == 1:
                return self.velocity_scaling_1()
            if stage_id == 2:
                self.velocity_scaling_2(**kwargs)
        elif thermostat_type == 'nose_hoover':
            raise ValueError('`nose_hoover` does not exist.')

    @logger_wraps()
    def velocity_scaling_1(self):
        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
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
        lmbd = (self.model.initial_temperature / temperature) ** 0.5

        debug_info(f'Kinetic Energy: {system_kinetic_energy};')
        debug_info(f'Temperature before velocity_scaling_1: {temperature};')
        debug_info(f'Initial Temperature: {self.model.initial_temperature};')
        debug_info(f'Lambda: {lmbd};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            vel_coefficient=lmbd,
        )
        debug_info(f'Temperature after velocity_scaling_1: {self.dynamic.temperature()};')
        return system_kinetic_energy, temperature

    @logger_wraps()
    def velocity_scaling_2(
            self,
            potential_energy: float,
            system_kinetic_energy: float,
            pressure: float,
    ):
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        debug_info(f'Pressure: {pressure};')
        debug_info(f'Temperature before velocity_scaling_2: {temperature};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
        total_energy = system_kinetic_energy + potential_energy
        debug_info(f'Temperature after velocity_scaling_2: {self.dynamic.temperature()};')
        return pressure, total_energy

    # @logger_wraps()
    # def nose_hoover_1(self):
    #     self.dynamic.get_next_positions(
    #         time_step=self.model.time_step,
    #         acc_coefficient=(lmbd + xi),
    #     )
    #     system_kinetic_energy = self.dynamic.system_kinetic_energy
    #     temperature = self.dynamic.temperature(
    #         system_kinetic_energy=system_kinetic_energy,
    #     )
    #
    #     self.model.initial_temperature += (
    #             sign(self.external.temperature - self.model.initial_temperature)
    #             * self.external.heating_velocity
    #             * self.model.time_step
    #     )
    #     if self.model.initial_temperature < TEMPERATURE_MINIMUM:
    #         self.model.initial_temperature = TEMPERATURE_MINIMUM
    #     if temperature <= 0:
    #         temperature = self.model.initial_temperature
    #
    #     diff = ((system_kinetic_energy
    #              - 3.0 * self.static.particles_number * self.model.initial_temperature)
    #             * self.model.time_step / self.external.thermostat_parameter)
    #     S_f += lmbd * self.model.time_step + diff * self.model.time_step
    #     lmbd += diff / 2.0
    #
    #     debug_info(f'Kinetic Energy: {system_kinetic_energy};')
    #     debug_info(f'Temperature before velocity_scaling_1: {temperature};')
    #     debug_info(f'Initial Temperature: {self.model.initial_temperature};')
    #     debug_info(f'Lambda: {lmbd};')
    #     debug_info(f'S_f: {S_f};')
    #     self.dynamic.get_next_velocities(
    #         time_step=self.model.time_step,
    #         acc_coefficient=(lmbd + xi),
    #     )
    #     debug_info(f'Temperature after velocity_scaling_1: {self.dynamic.temperature()};')
    #     return system_kinetic_energy, temperature
    #
    # @logger_wraps()
    # def nose_hoover_2(
    #         self,
    #         potential_energy: float,
    #         system_kinetic_energy: float,
    #         pressure: float,
    # ):
    #     cell_volume = self.static.get_cell_volume()
    #     density = self.static.get_density()
    #     pressure = self.dynamic.get_pressure(
    #         virial=virial,
    #         temperature=temperature,
    #         cell_volume=cell_volume,
    #     )
    #     xi += (pressure - P_ext) * self.model.time_step / self.external.barostat_parameter / density
    #     self.static.cell_dimensions *= 1.0 + xi * self.model.time_step
    #     cell_volume = self.static.get_cell_volume()
    #     density = self.static.get_density()
    #     error = 1e-5
    #     system_kinetic_energy = self.dynamic.system_kinetic_energy
    #     p_s = lmbd
    #     ready = False
    #     iter = 0

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
