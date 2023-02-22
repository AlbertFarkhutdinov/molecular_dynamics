import numpy as np

from scripts.integrators.base_integrator import BaseIntegrator
from scripts.log_config import logger_wraps, log_debug_info


class MTTKNVT(BaseIntegrator):

    def __init__(self, **integrator_kwargs):
        log_debug_info(f'{self.__class__.__name__} instance initialization.')
        super().__init__(**integrator_kwargs)
        self.scale = 1.0
        self.time_step_divider = 1
        self.weights_size = 3
        self.weights = self._get_weights(weights_size=self.weights_size)

        self.thermostats_number = self.external.thermostat_parameters.size
        self.thermostat_velocities = np.zeros(self.thermostats_number)
        self.thermostat_positions = np.zeros(self.thermostats_number)
        self.forces = np.zeros(self.thermostats_number, dtype=np.float)
        # self._init_integrator_parameters()
        log_debug_info(f'self.thermostats_number = {self.thermostats_number}')

    def _init_integrator_parameters(self):
        self._update_forces_first()
        for k in range(1, self.thermostats_number):
            self.forces[k] = self._update_forces_not_first(k=k)
        log_debug_info(f'thermostat_velocities = {self.thermostat_velocities}')
        log_debug_info(f'thermostat_positions = {self.thermostat_positions}')
        log_debug_info(f'forces = {self.forces}')

    @staticmethod
    @logger_wraps()
    def _get_weights(weights_size: int = 1):
        weights = np.array([1.0])
        if weights_size == 3:
            weights = 1 / (2 - 2 ** (1 / 3)) * np.ones(3)
            weights[1] = 1 - 2 * weights[0]
        if weights_size == 5:
            weights = 1 / (4 - 4 ** (1 / 3)) * np.ones(5)
            weights[2] = 1 - 4 * weights[0]
        return weights

    @logger_wraps()
    def _update_forces_first(self):
        self.forces[0] = (
                                2 * self.system.configuration.kinetic_energy
                                * self.scale ** 2
                                - 3 * self.external.temperature
                                * self.system.configuration.particles_number
                        ) / self.external.thermostat_parameters[0]

    @logger_wraps()
    def _update_forces_not_first(self, thermostat_number: int):
        _k = thermostat_number
        self.forces[_k] = (
                                 self.external.thermostat_parameters[_k - 1]
                                 * self.thermostat_velocities[_k - 1]
                                 * self.thermostat_velocities[_k - 1]
                                 - self.external.temperature
                         ) / self.external.thermostat_parameters[_k]

    @logger_wraps()
    def _update_thermostat_velocities_last(self, step: float):
        self.thermostat_velocities[self.thermostats_number - 1] += (
                self.forces[self.thermostats_number - 1] * step / 4
        )

    @logger_wraps()
    def _update_thermostat_velocities_not_last(
            self,
            step: float,
            is_right: bool,
    ):
        for j in range(self.thermostats_number - 1):
            k = self.thermostats_number - j if is_right else j
            _factor = np.exp(-self.thermostat_velocities[k + 1] * step / 8)
            self.thermostat_velocities[k] *= _factor
            self.thermostat_velocities[k] += self.forces[k] * step / 4
            self.thermostat_velocities[k] *= _factor
            if not is_right:
                self.forces[k + 1] = self._update_forces_not_first(k + 1)
        return self.forces

    @logger_wraps()
    def _update_particle_velocities(self, step: float):
        _factor = np.exp(-step / 2 * self.thermostat_velocities[0])
        self.scale *= _factor

    @logger_wraps()
    def _update_thermostat_positions(self, step: float):
        for j in range(self.thermostats_number):
            self.thermostat_positions[j] += (
                    self.thermostat_velocities[j] * step / 2
            )

    @logger_wraps()
    def do_chain(self):
        self.scale = 1.0
        self._update_forces_first()
        for _ in range(self.time_step_divider):
            for i in range(self.weights_size):
                _step = (
                        self.weights[i] * self.time_step
                        / self.time_step_divider
                )
                self._update_thermostat_velocities_last(step=_step)
                self._update_thermostat_velocities_not_last(
                    step=_step,
                    is_right=True,
                )
                self._update_particle_velocities(step=_step)
                self._update_forces_first()
                self._update_thermostat_positions(step=_step)
                self._update_thermostat_velocities_not_last(
                    step=_step,
                    is_right=False,
                )
                self._update_thermostat_velocities_last(step=_step)
        self.system.configuration.velocities *= self.scale

    @logger_wraps()
    def stage_1(self):
        if self.thermostats_number > 0:
            self.do_chain()
        self.get_next_velocities()
        self.get_next_positions()

    @logger_wraps()
    def stage_2(self):
        self.get_next_velocities()
        if self.thermostats_number > 0:
            self.do_chain()
