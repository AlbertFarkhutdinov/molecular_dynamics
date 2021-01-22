from scripts.constants import TEMPERATURE_MINIMUM
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import sign
from scripts.log_config import log_debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters


class VelocityScaling:

    def __init__(
            self,
            dynamic: SystemDynamicParameters,
            external: ExternalParameters,
            model: ModelingParameters,
    ):
        self.dynamic = dynamic
        self.external = external
        self.model = model
        self.nvt_factor = 0.0

    @logger_wraps()
    def stage_1(self):
        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
        )
        temperature = self.dynamic.temperature()
        self.model.initial_temperature += (
                sign(self.external.temperature - self.model.initial_temperature)
                * self.external.heating_velocity
                * self.model.time_step
        )
        if self.model.initial_temperature < TEMPERATURE_MINIMUM:
            self.model.initial_temperature = TEMPERATURE_MINIMUM
        if temperature <= 0:
            temperature = self.model.initial_temperature

        self.nvt_factor = (self.model.initial_temperature / temperature) ** 0.5

        log_debug_info(f'Initial Temperature: {self.model.initial_temperature};')
        log_debug_info(f'NVT-factor: {self.nvt_factor};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            vel_coefficient=self.nvt_factor,
        )

    @logger_wraps()
    def stage_2(self):
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
