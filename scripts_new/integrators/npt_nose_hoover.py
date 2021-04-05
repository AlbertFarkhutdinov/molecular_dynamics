# from scripts_new.constants import TEMPERATURE_MINIMUM
# from scripts_new.integrators.base_integrator import BaseIntegrator
# from scripts_new.helpers import sign
# from scripts_new.log_config import log_debug_info, logger_wraps
#
#
# class NoseHooverNPT(BaseIntegrator):
#
#     def __init__(
#             self,
#             **integrator_kwargs,
#     ):
#         super().__init__(**integrator_kwargs)
#         self.npt_factor = 0.0
#         self.nvt_factor = 0.0
#         self.s_f = 0.0
#
#     @logger_wraps()
#     def stage_1(self):
#         acc_coefficient = self.nvt_factor + self.npt_factor
#         self.get_next_positions(
#             acc_factor=acc_coefficient,
#         )
#         kinetic_energy = self.system.kinetic_energy
#         self.model.initial_temperature += (
#                 sign(self.external.temperature - self.model.initial_temperature)
#                 * self.external.heating_velocity
#                 * self.time_step
#         )
#         if self.model.initial_temperature < TEMPERATURE_MINIMUM:
#             self.model.initial_temperature = TEMPERATURE_MINIMUM
#         lambda_diff = (
#                 (2.0 * kinetic_energy
#                  - 3.0 * self.system.particles_number
#                  * self.model.initial_temperature)
#                 * self.time_step
#                 / self.external.thermostat_parameters[0]
#         )
#         self.s_f += (
#                 self.nvt_factor * self.time_step
#                 + lambda_diff * self.time_step
#         )
#         self.nvt_factor += lambda_diff / 2.0
#         log_debug_info(
#             f'Initial Temperature: {self.model.initial_temperature};'
#         )
#         log_debug_info(f'NVT-factor: {self.nvt_factor};')
#         self.get_next_velocities()
#
#     @logger_wraps()
#     def stage_2(
#             self,
#             potential_energy: float,
#             kinetic_energy: float,
#     ):
#         temperature = self.system.get_temperature(
#             kinetic_energy=kinetic_energy,
#         )
#         volume = self.system.volume
#         density = self.system.get_density(volume=volume)
#         pressure = self.system.get_pressure(
#             temperature=temperature,
#             volume=volume,
#             density=density,
#         )
#         log_debug_info(f'Cell Volume before nose_hoover_2: {volume};')
#         log_debug_info(f'Density before nose_hoover_2: {density};')
#         log_debug_info(f'Temperature before nose_hoover_2: {temperature};')
#         log_debug_info(f'Pressure before nose_hoover_2: {pressure};')
#         self.npt_factor += (
#                 (pressure - self.external.pressure)
#                 * self.time_step
#                 / self.external.barostat_parameter / density
#         )
#         log_debug_info(f'NPT-factor: {self.npt_factor};')
#         self.system.cell_dimensions *= 1.0 + self.npt_factor * self.time_step
#
#         volume = self.system.volume
#         density = self.system.get_density(volume=volume)
#         pressure = self.system.get_pressure(
#             temperature=temperature,
#             volume=volume,
#             density=density,
#         )
#         log_debug_info(f'New cell dimensions: {self.system.cell_dimensions};')
#         log_debug_info(f'New cell volume: {volume};')
#         log_debug_info(f'New density: {density};')
#         log_debug_info(f'Pressure at new volume: {pressure};')
#
#         # TODO incorrect algorithm
#         error = 1e-5
#         half_time = self.time_step / 2.0
#         time_ratio = self.time_step / self.external.thermostat_parameters[0]
#         kinetic_energy = self.system.kinetic_energy
#         nvt_factor_new = self.nvt_factor
#         velocities_new = self.system.velocities
#         is_ready = False
#         for i in range(50):
#             log_debug_info(f'i: {i}')
#             if is_ready:
#                 break
#             nvt_factor_old = nvt_factor_new
#             ds = 0
#             velocities_old = velocities_new
#             b = (
#                     -half_time
#                     * (
#                             self.system.accelerations
#                             - nvt_factor_old * velocities_new
#                     )
#                     - (self.system.velocities - velocities_old)
#             )
#             ds += (velocities_old * b).sum() * time_ratio
#             dd = 1 - nvt_factor_old * half_time
#             ds -= dd * (
#                     (
#                             3.0 * self.system.particles_number
#                             * self.model.initial_temperature
#                             - 2.0 * kinetic_energy
#                     )
#                     * time_ratio / 2.0
#                     - (self.nvt_factor - nvt_factor_old)
#             )
#             ds = ds / (
#                     -self.time_step * kinetic_energy * time_ratio + dd
#             )
#             velocities_new += (b + half_time * velocities_old * ds) / dd
#             kinetic_energy = (
#                                             velocities_new * velocities_new
#                                     ).sum() / 2.0
#             nvt_factor_new = nvt_factor_old + ds
#
#             is_ready = (
#                     (abs((velocities_new - velocities_old)
#                          / velocities_new) <= error).all()
#                     and (abs((nvt_factor_new - nvt_factor_old) / nvt_factor_new) <= error).all()
#             )
#
#         self.system.velocities = velocities_new
#         self.nvt_factor = nvt_factor_new
#         total_energy = (
#                 kinetic_energy + potential_energy
#                 + self.nvt_factor * self.nvt_factor
#                 * self.external.thermostat_parameters[0] / 2.0
#                 + 3.0 * self.system.particles_number
#                 * self.model.initial_temperature * self.s_f
#         )
