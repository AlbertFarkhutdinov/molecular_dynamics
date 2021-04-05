from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.log_config import logger_wraps
from scripts.modeling_parameters import ModelingParameters


class NVE:

    def __init__(
            self,
            dynamic: SystemDynamicParameters,
            model: ModelingParameters,
    ):
        self.dynamic = dynamic
        self.model = model

    @logger_wraps()
    def stage_1(self):
        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
        )
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )

    @logger_wraps()
    def stage_2(self):
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
