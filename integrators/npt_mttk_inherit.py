import numpy as np

from integrators.nvt_mttk import IsothermMTTK


class IsobarMTTK(IsothermMTTK):

    def __init__(self, **integrator_kwargs):
        self.logger.debug(f'{self.__class__.__name__} instance initialization.')
        super().__init__(**integrator_kwargs)
        self.kinetic_energy = self.system.configuration.kinetic_energy
        self.volume = self.system.volume
        self.internal_pressure = self.system.get_pressure()
        self.epsilon = np.log(self.system.volume / 3)
        self.bar_velocity = 0.0
        self.bar_force = 0.0
        self.logger.debug(f'self.epsilon = {self.epsilon}')

    def _update_bar_force(self):
        self.bar_force = (
                             (1 + self.system.configuration.particles_number)
                             / self.system.configuration.particles_number
                             * 2 * self.kinetic_energy
                             + 3.0 * self.volume
                             * (
                                     self.internal_pressure
                                     - self.external.pressure
                             )
                     ) / self.external.barostat_parameter

    def _update_particle_velocities(self, step: float):
        _factor = np.exp(-step / 2 * self.therm_velocities[0])
        _factor *= np.exp(
            -step / 2
            * (1 + 1 / self.system.configuration.particles_number)
            * self.bar_velocity
        )
        self.scale *= _factor
        self.kinetic_energy *= _factor ** 2
        self._update_bar_force()

    def _update_bar_velocity(self, step: float):
        _factor = np.exp(-step / 8 * self.therm_velocities[0])
        self.bar_velocity *= _factor
        self.bar_velocity += self.bar_force * step / 4
        self.bar_velocity *= _factor

    @staticmethod
    def __get_poly(arg: float) -> float:
        e_2 = 1.0 / 6.0
        e_4 = e_2 / 20.0
        e_6 = e_4 / 42.0
        e_8 = e_6 / 72.0
        return (((e_8 * arg + e_6) * arg + e_4) * arg + e_2) * arg + 1.0

    def do_chain(self):
        self.kinetic_energy = self.system.configuration.kinetic_energy
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
                self._update_bar_velocity(step=_step)
                self._update_particle_velocities(step=_step)
                self._update_therm_positions(step=_step)
                self._update_bar_velocity(step=_step)
                self._update_forces_first()
                self._update_therm_velocities_not_last(
                    step=_step,
                    is_right=False,
                )
                self._update_therm_velocities_last(step=_step)
        self.system.configuration.velocities *= self.scale

    def stage_1(self):
        super().stage_1()
        v_log_v_dt2 = self.time_step ** 2 * self.bar_velocity
        _a = np.exp(v_log_v_dt2)
        poly = self.__get_poly(v_log_v_dt2 ** 2)
        _b = _a * poly * self.time_step
        self.system.configuration.positions *= _a ** 2
        self.system.configuration.positions += (
                self.system.configuration.velocities * _b
        )
        self.epsilon += self.bar_velocity * self.time_step
        # self.system.cell_dimensions = (
        #         np.ones(3) * (3 * np.exp(self.epsilon)) ** (1 / 3)
        # )
