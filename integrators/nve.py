from integrators.base_integrator import BaseIntegrator


class NVE(BaseIntegrator):

    def stage_1(self):
        self.get_next_positions()
        self.get_next_velocities()

    def stage_2(self):
        self.get_next_velocities()
