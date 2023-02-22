import numpy as np

from integrators.base_integrator import BaseIntegrator


class IsothermMTTK(BaseIntegrator):

    def __init__(self, **integrator_kwargs):
        self.logger.debug(f'{self.__class__.__name__} instance initialization.')
        super().__init__(**integrator_kwargs)
        self.scale = 1.0
        self.time_step_divider = 1
        self.weights_size = 3
        self.weights = self._get_weights(weights_size=self.weights_size)
        self.therm_number = self.external.thermostat_parameters.size
        self.therm_velocities = np.zeros(self.therm_number)
        self.therm_positions = np.zeros(self.therm_number)
        self.therm_forces = np.zeros(self.therm_number, dtype=np.float)
        self.logger.debug(f'Thermostats number = {self.therm_number}')

    @staticmethod
    def _get_weights(weights_size: int = 1):
        weights = np.array([1.0])
        if weights_size == 3:
            weights = 1 / (2 - 2 ** (1 / 3)) * np.ones(3)
            weights[1] = 1 - 2 * weights[0]
        if weights_size == 5:
            weights = 1 / (4 - 4 ** (1 / 3)) * np.ones(5)
            weights[2] = 1 - 4 * weights[0]
        return weights

    def _update_forces_first(self):
        self.therm_forces[0] = (
                                2 * self.system.configuration.kinetic_energy
                                * self.scale ** 2
                                - 3 * self.external.temperature
                                * self.system.configuration.particles_number
                        ) / self.external.thermostat_parameters[0]

    def _update_forces_not_first(self, therm_number: int):
        _k = therm_number
        self.therm_forces[_k] = (
                                 self.external.thermostat_parameters[_k - 1]
                                 * self.therm_velocities[_k - 1] ** 2
                                 - self.external.temperature
                         ) / self.external.thermostat_parameters[_k]

    def _update_therm_velocities_last(self, step: float):
        self.therm_velocities[self.therm_number - 1] += (
                self.therm_forces[self.therm_number - 1] * step / 4
        )

    def _update_therm_velocities_not_last(
            self,
            step: float,
            is_right: bool,
    ) -> np.ndarray:
        for j in range(self.therm_number - 1):
            k = self.therm_number - j if is_right else j
            _factor = np.exp(-self.therm_velocities[k + 1] * step / 8)
            self.therm_velocities[k] *= _factor
            self.therm_velocities[k] += self.therm_forces[k] * step / 4
            self.therm_velocities[k] *= _factor
            if not is_right:
                self.therm_forces[k + 1] = self._update_forces_not_first(k + 1)
        return self.therm_forces

    def _update_particle_velocities(self, step: float):
        _factor = np.exp(-step / 2 * self.therm_velocities[0])
        self.scale *= _factor

    def _update_therm_positions(self, step: float):
        for j in range(self.therm_number):
            self.therm_positions[j] += self.therm_velocities[j] * step / 2

    def do_chain(self):
        self.scale = 1.0
        self._update_forces_first()
        for _ in range(self.time_step_divider):
            for i in range(self.weights_size):
                _step = (
                        self.weights[i] * self.time_step
                        / self.time_step_divider
                )
                self._update_therm_velocities_last(step=_step)
                self._update_therm_velocities_not_last(
                    step=_step,
                    is_right=True,
                )
                self._update_particle_velocities(step=_step)
                self._update_forces_first()
                self._update_therm_positions(step=_step)
                self._update_therm_velocities_not_last(
                    step=_step,
                    is_right=False,
                )
                self._update_therm_velocities_last(step=_step)
        self.system.configuration.velocities *= self.scale

    def stage_1(self):
        if self.therm_number > 0:
            self.do_chain()
        self.get_next_velocities()
        self.get_next_positions()

    def stage_2(self):
        self.get_next_velocities()
        if self.therm_number > 0:
            self.do_chain()
