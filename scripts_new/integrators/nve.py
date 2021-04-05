from scripts_new.integrators.base_integrator import BaseIntegrator
from scripts_new.log_config import logger_wraps


class NVE(BaseIntegrator):

    @logger_wraps()
    def stage_1(self):
        self.get_next_positions()
        self.get_next_velocities()

    @logger_wraps()
    def stage_2(self):
        self.get_next_velocities()
