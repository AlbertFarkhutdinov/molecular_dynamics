import numpy as np

from scripts.integrators.base_integrator import BaseIntegrator
from scripts.log_config import logger_wraps, log_debug_info


class MTTK(BaseIntegrator):

    def __init__(self, **integrator_kwargs):
        log_debug_info(f'{self.__class__.__name__} instance initialization.')
        super().__init__(**integrator_kwargs)
        self.npt_factor = 0.0
        self.nvt_factors = np.zeros(
            self.external.thermostat_parameters.size,
            dtype=np.float,
        )
        self.xis = np.zeros(
            self.external.thermostat_parameters.size,
            dtype=np.float,
        )
        self.epsilon = np.log(self.system.volume / 3)
        self.thermostats_number = self.external.thermostat_parameters.size

        kinetic_energy = self.system.configuration.kinetic_energy
        self.g_npt = self.get_g_npt(
            kinetic_energy=kinetic_energy,
        )
        self.g_nvt = np.zeros(self.thermostats_number, dtype=np.float)
        self.g_nvt[0] = self.get_g_nvt_0(
            kinetic_energy=kinetic_energy,
        )
        for k in range(1, self.thermostats_number):
            self.g_nvt[k] = self.get_g_nvt_k(k=k)
        log_debug_info(f'self.npt_factor = {self.npt_factor}')
        log_debug_info(f'self.nvt_factors = {self.nvt_factors}')
        log_debug_info(f'self.xis = {self.xis}')
        log_debug_info(f'self.epsilon = {self.epsilon}')
        log_debug_info(f'self.thermostats_number = {self.thermostats_number}')
        log_debug_info(f'system_kinetic_energy = {kinetic_energy}')
        log_debug_info(f'self.g_npt = {self.g_npt}')
        log_debug_info(f'self.g_nvt = {self.g_nvt}')
        log_debug_info(f'positions mean = {self.system.configuration.positions.mean()}')
        log_debug_info(f'velocities mean = {self.system.configuration.velocities.mean()}')
        log_debug_info(f'accelerations mean = {self.system.configuration.accelerations.mean()}')

    @logger_wraps()
    def stage_1(self):
        self.nose_hoover_chain()
        fact_3 = 6
        fact_5 = fact_3 * 20
        fact_7 = fact_5 * 42
        fact_9 = fact_7 * 72
        self.get_next_velocities()
        vlogv_dt2 = self.time_step ** 2 * self.npt_factor
        _a = np.exp(vlogv_dt2)
        _a_2 = _a * _a
        arg_2 = vlogv_dt2 * vlogv_dt2
        poly = (
                1.0
                + arg_2 / fact_3
                + arg_2 ** 2 / fact_5
                + arg_2 ** 3 / fact_7
                + arg_2 ** 4 / fact_9
        )
        _b = _a * poly * self.time_step
        log_debug_info(
            f'positions mean = {self.system.configuration.positions.mean()}'
        )
        self.system.configuration.positions *= _a_2
        log_debug_info(
            f'positions mean = {self.system.configuration.positions.mean()}'
        )

        self.system.configuration.positions += (
                self.system.configuration.velocities * _b
        )
        log_debug_info(
            f'positions mean = {self.system.configuration.positions.mean()}'
        )
        self.epsilon += self.npt_factor * self.time_step
        self.system.cell_dimensions = (
                np.ones(3) * (3 * np.exp(self.epsilon)) ** (1 / 3)
        )

    @logger_wraps()
    def stage_2(self):
        self.get_next_velocities()
        self.nose_hoover_chain()

    @logger_wraps()
    def nose_hoover_chain(self):
        scale, time_step_divider, n_ys = 1.0, 1, 3
        _w = self.get_w(n_ys=n_ys)

        for _ in range(time_step_divider):
            for i in range(n_ys):
                _step = _w[i] * self.time_step / time_step_divider
                self.do_liouville_nvt_last(
                    step=_step,
                )
                self.do_liouville_nvt_not_last(
                    step=_step,
                    is_right=True,
                )
                self.do_liouville_npt(
                    step=_step,
                )
                scale, _ = self.do_liouville_velocities(
                    step=_step,
                    scale=scale,
                )
                self.do_liouville_xis(
                    step=_step,
                )
                self.do_liouville_npt(
                    step=_step,
                )
                self.do_liouville_nvt_not_last(
                    step=_step,
                    is_right=False,
                )
                self.do_liouville_nvt_last(
                    step=_step,
                )

    @logger_wraps()
    def get_g_nvt_0(self, kinetic_energy: float):
        return (
                       2 * kinetic_energy
                       + self.external.barostat_parameter
                       * self.npt_factor * self.npt_factor
                       - (3 * self.system.configuration.particles_number + 1)
                       * self.external.temperature
               ) / self.external.thermostat_parameters[0]

    @logger_wraps()
    def get_g_nvt_k(self, k: int):
        return (
                       self.external.thermostat_parameters[k - 1]
                       * self.nvt_factors[k - 1]
                       * self.nvt_factors[k - 1]
                       - self.external.temperature
               ) / self.external.thermostat_parameters[k]

    @logger_wraps()
    def get_g_npt(
            self,
            kinetic_energy: float,
    ):
        volume = self.system.volume
        density = self.system.get_density(volume=volume)
        temperature = self.system.configuration.get_temperature()
        internal_pressure = self.system.get_pressure(
            volume=volume,
            density=density,
            temperature=temperature,
        )
        log_debug_info(f'volume = {volume}')
        log_debug_info(f'density = {density}')
        log_debug_info(f'temperature = {temperature}')
        log_debug_info(f'internal_pressure = {internal_pressure}')
        return (
                       1 / self.system.configuration.particles_number
                       * 2 * kinetic_energy
                       + 3.0 * volume
                       * (internal_pressure - self.external.pressure)
               ) / self.external.barostat_parameter

    @logger_wraps()
    def do_liouville_nvt_last(
            self,
            step: float,
    ):
        self.nvt_factors[
            self.thermostats_number - 1
        ] += self.g_nvt[self.thermostats_number - 1] * step / 4

    @logger_wraps()
    def do_liouville_nvt_not_last(
            self,
            step: float,
            is_right: bool,
    ):
        for j in range(self.thermostats_number - 1):
            k = self.thermostats_number - 2 - j if is_right else j
            _factor = np.exp(self.nvt_factors[k + 1] * step / 8)
            self.nvt_factors[k] *= _factor
            self.nvt_factors[k] += self.g_nvt[k] * step / 4
            self.nvt_factors[k] *= _factor
            if not is_right:
                self.g_nvt[j + 1] = self.get_g_nvt_k(k=j + 1)
        return self.g_nvt

    @logger_wraps()
    def do_liouville_npt(
            self,
            step: float,
    ):
        _factor = np.exp(-step / 8 * self.nvt_factors[0])
        self.npt_factor *= _factor
        self.npt_factor += self.g_npt * step / 4
        self.npt_factor *= _factor

    @logger_wraps()
    def do_liouville_velocities(
            self,
            step: float,
            scale: float,
    ):
        _factor = np.exp(
            -step / 2 * (
                    self.nvt_factors[0]
                    + (1 + 1 / self.system.configuration.particles_number)
                    * self.npt_factor
            )
        )
        scale *= _factor
        self.system.configuration.velocities *= scale
        kinetic_energy = self.system.configuration.kinetic_energy
        self.g_npt = self.get_g_npt(
            kinetic_energy=kinetic_energy,
        )
        self.g_nvt[0] = self.get_g_nvt_0(
            kinetic_energy=kinetic_energy,
        )
        return scale, kinetic_energy

    @logger_wraps()
    def do_liouville_xis(
            self,
            step: float,
    ):
        for j in range(self.thermostats_number):
            self.xis[j] += self.nvt_factors[j] * step / 2

    @staticmethod
    @logger_wraps()
    def get_w(n_ys: int = 1):
        _w = np.array([1.0])
        if n_ys == 3:
            _w = 1 / (2 - 2 ** (1 / 3)) * np.ones(3)
            _w[1] = 1 - 2 * _w[0]
        if n_ys == 5:
            _w = 1 / (4 - 4 ** (1 / 3)) * np.ones(5)
            _w[2] = 1 - 4 * _w[0]
        return _w
