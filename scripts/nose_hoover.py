from scripts.constants import TEMPERATURE_MINIMUM
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import sign
from scripts.log_config import log_debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters
from scripts.potential_parameters import PotentialParameters
from scripts.static_parameters import SystemStaticParameters


class NoseHooverNPT:

    def __init__(
            self,
            static: SystemStaticParameters,
            dynamic: SystemDynamicParameters,
            external: ExternalParameters,
            model: ModelingParameters,
    ):
        self.static = static
        self.dynamic = dynamic
        self.external = external
        self.model = model
        self.npt_factor = 0.0
        self.nvt_factor = 0.0
        self.s_f = 0.0

    @logger_wraps()
    def stage_1(self):
        acc_coefficient = self.nvt_factor + self.npt_factor
        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
            acc_coefficient=acc_coefficient,
        )
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        self.model.initial_temperature += (
                sign(self.external.temperature - self.model.initial_temperature)
                * self.external.heating_velocity
                * self.model.time_step
        )
        if self.model.initial_temperature < TEMPERATURE_MINIMUM:
            self.model.initial_temperature = TEMPERATURE_MINIMUM
        lambda_diff = (
                (2.0 * system_kinetic_energy
                 - 3.0 * self.static.particles_number
                 * self.model.initial_temperature)
                * self.model.time_step
                / self.external.thermostat_parameters[0]
        )
        self.s_f += self.nvt_factor * self.model.time_step + lambda_diff * self.model.time_step
        self.nvt_factor += lambda_diff / 2.0
        log_debug_info(f'Initial Temperature: {self.model.initial_temperature};')
        log_debug_info(f'NVT-factor: {self.nvt_factor};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )

    @logger_wraps()
    def stage_2(
            self,
            potential_energy: float,
            system_kinetic_energy: float,
    ):
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        cell_volume = self.static.get_cell_volume()
        density = self.static.get_density()
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            cell_volume=cell_volume,
        )
        log_debug_info(f'Cell Volume before nose_hoover_2: {cell_volume};')
        log_debug_info(f'Density before nose_hoover_2: {density};')
        log_debug_info(f'Temperature before nose_hoover_2: {temperature};')
        log_debug_info(f'Pressure before nose_hoover_2: {pressure};')
        self.npt_factor += (
                (pressure - self.external.pressure)
                * self.model.time_step
                / self.external.barostat_parameter / density
        )
        log_debug_info(f'NPT-factor: {self.npt_factor};')
        self.static.cell_dimensions *= 1.0 + self.npt_factor * self.model.time_step

        cell_volume = self.static.get_cell_volume()
        density = self.static.get_density()
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            cell_volume=cell_volume,
        )
        log_debug_info(f'New cell dimensions: {self.static.cell_dimensions};')
        log_debug_info(f'New cell volume: {cell_volume};')
        log_debug_info(f'New density: {density};')
        log_debug_info(f'Pressure at new volume: {pressure};')

        # TODO incorrect algorithm
        error = 1e-5
        half_time = self.model.time_step / 2.0
        time_ratio = self.model.time_step / self.external.thermostat_parameters[0]
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        nvt_factor_new = self.nvt_factor
        velocities_new = self.dynamic.velocities
        is_ready = False
        for i in range(50):
            log_debug_info(f'i: {i}')
            if is_ready:
                break
            nvt_factor_old = nvt_factor_new
            ds = 0
            velocities_old = velocities_new
            b = (
                    -half_time
                    * (self.dynamic.accelerations - nvt_factor_old * velocities_new)
                    - (self.dynamic.velocities - velocities_old)
            )
            ds += (velocities_old * b).sum() * time_ratio
            dd = 1 - nvt_factor_old * half_time
            ds -= dd * (
                    (
                            3.0 * self.static.particles_number
                            * self.model.initial_temperature
                            - 2.0 * system_kinetic_energy
                    )
                    * time_ratio / 2.0
                    - (self.nvt_factor - nvt_factor_old)
            )
            ds = ds / (-self.model.time_step * system_kinetic_energy * time_ratio + dd)
            velocities_new += (b + half_time * velocities_old * ds) / dd
            system_kinetic_energy = (velocities_new * velocities_new).sum() / 2.0
            nvt_factor_new = nvt_factor_old + ds

            is_ready = (
                    (abs((velocities_new - velocities_old) / velocities_new) <= error).all()
                    and (abs((nvt_factor_new - nvt_factor_old) / nvt_factor_new) <= error).all()
            )

        self.dynamic.velocities = velocities_new
        self.nvt_factor = nvt_factor_new
        total_energy = (
                system_kinetic_energy + potential_energy
                + self.nvt_factor * self.nvt_factor * self.external.thermostat_parameters[0] / 2.0
                + 3.0 * self.static.particles_number * self.model.initial_temperature * self.s_f
        )

