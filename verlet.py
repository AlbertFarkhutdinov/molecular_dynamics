import numpy as np

from dynamic_parameters import SystemDynamicParameters
from external_parameters import ExternalParameters
from helpers import get_empty_float_scalars, get_empty_int_scalars, get_empty_vectors, sign
from log_config import debug_info, logger_wraps
from modeling_parameters import ModelingParameters
from numba_procedures import lf_cycle, update_list_cycle
from potential_parameters import PotentialParameters
from static_parameters import SystemStaticParameters


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
        self.verlet_lists = {
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
        if temperature == 0:
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
            verlet_list=self.verlet_lists['list'],
            marker_1=self.verlet_lists['marker_1'],
            marker_2=self.verlet_lists['marker_2'],
            r_cut=self.potential.r_cut,
            potential_table=potential_table,
            potential_energies=potential_energies,
            positions=self.dynamic.positions,
            accelerations=self.dynamic.accelerations,
            cell_dimensions=self.static.cell_dimensions,
        )
        acc_mag = (self.dynamic.accelerations ** 2).sum(axis=1) ** 0.2
        debug_info(f'Mean and max acceleration: {acc_mag.mean()}, {acc_mag.max()}')
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
        self.verlet_lists = {
            'marker_1': get_empty_int_scalars(self.static.particles_number),
            'marker_2': get_empty_int_scalars(self.static.particles_number),
            'list': get_empty_int_scalars(100 * self.static.particles_number),
        }
        advances = get_empty_int_scalars(self.static.particles_number)
        update_list_cycle(
            rng=self.potential.r_cut + self.potential.skin,
            advances=advances,
            particles_number=self.static.particles_number,
            positions=self.dynamic.positions,
            cell_dimensions=self.static.cell_dimensions,
            marker_1=self.verlet_lists['marker_1'],
            marker_2=self.verlet_lists['marker_2'],
            verlet_list=self.verlet_lists['list'],
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