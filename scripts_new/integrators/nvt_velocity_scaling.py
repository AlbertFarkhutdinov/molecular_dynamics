from scripts_new.constants import TEMPERATURE_MINIMUM
from scripts_new.integrators.base_integrator import BaseIntegrator
from scripts_new.helpers import sign
from scripts_new.log_config import log_debug_info, logger_wraps


class VelocityScaling(BaseIntegrator):

    def __init__(self, **integrator_kwargs):
        super().__init__(**integrator_kwargs)
        self.nvt_factor = 0.0

    @logger_wraps()
    def stage_1(self):
        self.get_next_positions()
        temperature = self.system.configuration.get_temperature()
        self.initial_temperature += (
                sign(self.external.temperature - self.initial_temperature)
                * self.external.heating_velocity
                * self.time_step
        )
        if self.initial_temperature < TEMPERATURE_MINIMUM:
            self.initial_temperature = TEMPERATURE_MINIMUM
        if temperature <= 0:
            temperature = self.initial_temperature

        self.nvt_factor = (self.initial_temperature / temperature) ** 0.5

        log_debug_info(f'Initial Temperature: {self.initial_temperature};')
        log_debug_info(f'NVT-factor: {self.nvt_factor};')
        self.get_next_velocities(vel_factor=self.nvt_factor)

    @logger_wraps()
    def stage_2(self):
        self.get_next_velocities()
