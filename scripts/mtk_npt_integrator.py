import numpy as np

from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.static_parameters import SystemStaticParameters


class MTK:

    def __init__(
            self,
            static: SystemStaticParameters,
            dynamic: SystemDynamicParameters,
            external: ExternalParameters,
            time_step: float,
    ):
        self.static = static
        self.dynamic = dynamic
        self.external = external
        self.time_step = time_step
        self.npt_factor = 0.0
        self.nvt_factors = np.zeros(self.external.thermostat_parameters.size, dtype=np.float)
        self.xis = np.zeros(self.external.thermostat_parameters.size, dtype=np.float)
        self.epsilon = np.log(self.static.get_cell_volume() / 3)
        self.thermostats_number = self.external.thermostat_parameters.size

        system_kinetic_energy = self.dynamic.system_kinetic_energy
        self.g_npt = self.get_g_npt(
            system_kinetic_energy=system_kinetic_energy,
        )
        self.g_nvt = np.zeros(self.thermostats_number, dtype=np.float)
        self.g_nvt[0] = self.get_g_nvt_0(
            system_kinetic_energy=system_kinetic_energy,
        )
        for k in range(1, self.thermostats_number):
            self.g_nvt[k] = self.get_g_nvt_k(k=k)

    def stage_1(self):
        self.nose_hoover_chain()
        fact_3 = 6
        fact_5 = fact_3 * 20
        fact_7 = fact_5 * 42
        fact_9 = fact_7 * 72
        self.dynamic.get_next_velocities(time_step=self.time_step)
        vlogv_dt2 = self.time_step ** 2 * self.npt_factor
        aa = np.exp(vlogv_dt2)
        aa_2 = aa * aa
        arg_2 = vlogv_dt2 * vlogv_dt2
        poly = 1.0 + arg_2 / fact_3 + arg_2 ** 2 / fact_5 + arg_2 ** 3 / fact_7 + arg_2 ** 4 / fact_9
        bb = aa * poly * self.time_step
        self.dynamic.positions = self.dynamic.positions * aa_2 + self.dynamic.velocities * bb
        self.epsilon += self.npt_factor * self.time_step
        self.static.cell_dimensions = np.ones(3) * (3 * np.exp(self.epsilon)) ** (1 / 3)

    def stage_2(self):
        self.dynamic.get_next_velocities(time_step=self.time_step)
        self.nose_hoover_chain()

    def nose_hoover_chain(self):
        # TODO check operators order and g_nvt calculation
        scale, time_step_divider, n_ys = 1.0, 1, 3
        w = self.get_w(n_ys=n_ys)

        for _ in range(time_step_divider):
            for i in range(n_ys):
                dt = w[i] * self.time_step / time_step_divider
                self.do_liouville_nvt_last(
                    dt=dt,
                )
                self.do_liouville_nvt_not_last(
                    dt=dt,
                    is_right=True,
                )
                self.do_liouville_npt(
                    dt=dt,
                )
                scale, system_kinetic_energy = self.do_liouville_velocities(
                    dt=dt,
                    scale=scale,
                )
                self.do_liouville_xis(
                    dt=dt,
                )
                self.do_liouville_npt(
                    dt=dt,
                )
                self.do_liouville_nvt_not_last(
                    dt=dt,
                    is_right=False,
                )
                self.do_liouville_nvt_last(
                    dt=dt,
                )

    def get_g_nvt_0(self, system_kinetic_energy: float):
        return (
                       2 * system_kinetic_energy
                       + self.external.barostat_parameter
                       * self.npt_factor * self.npt_factor
                       - (3 * self.static.particles_number + 1)
                       * self.external.temperature
               ) / self.external.thermostat_parameters[0]

    def get_g_nvt_k(self, k: int):
        return (
                       self.external.thermostat_parameters[k - 1]
                       * self.nvt_factors[k - 1]
                       * self.nvt_factors[k - 1]
                       - self.external.temperature
               ) / self.external.thermostat_parameters[k]

    def get_g_npt(
            self,
            system_kinetic_energy: float,
    ):
        # TODO get right pressure and volume values.
        cell_volume = self.static.get_cell_volume()
        internal_pressure = self.dynamic.get_pressure(
            cell_volume=cell_volume,
            density=self.static.get_density(volume=cell_volume),
        )
        return (
                       1 / self.static.particles_number * 2 * system_kinetic_energy
                       + 3.0 * self.static.get_cell_volume() * (internal_pressure - self.external.pressure)
               ) / self.external.barostat_parameter

    def do_liouville_nvt_last(
            self,
            dt: float,
    ):
        self.nvt_factors[self.thermostats_number - 1] += self.g_nvt[self.thermostats_number - 1] * dt / 4

    def do_liouville_nvt_not_last(
            self,
            dt: float,
            is_right: bool,
    ):
        for j in range(self.thermostats_number - 1):
            k = self.thermostats_number - 2 - j if is_right else j
            _factor = np.exp(self.nvt_factors[k + 1] * dt / 8)
            self.nvt_factors[k] *= _factor
            self.nvt_factors[k] += self.g_nvt[k] * dt / 4
            self.nvt_factors[k] *= _factor
            if not is_right:
                self.g_nvt[j + 1] = self.get_g_nvt_k(k=j + 1)
        return self.g_nvt

    def do_liouville_npt(
            self,
            dt: float,
    ):
        _factor = np.exp(-dt / 8 * self.nvt_factors[0])
        self.npt_factor *= _factor
        self.npt_factor += self.g_npt * dt / 4
        self.npt_factor *= _factor

    def do_liouville_velocities(
            self,
            dt: float,
            scale: float,
    ):
        _factor = np.exp(
            -dt / 2 * (
                    self.nvt_factors[0]
                    + (1 + 1 / self.static.particles_number)
                    * self.npt_factor
            )
        )
        scale *= _factor
        self.dynamic.velocities *= scale
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        self.g_npt = self.get_g_npt(system_kinetic_energy=system_kinetic_energy)
        self.g_nvt[0] = self.get_g_nvt_0(system_kinetic_energy=system_kinetic_energy)
        return scale, system_kinetic_energy

    def do_liouville_xis(
            self,
            dt: float,
    ):
        for j in range(self.thermostats_number):
            self.xis[j] += self.nvt_factors[j] * dt / 2

    @staticmethod
    def get_w(n_ys: int = 1):
        w = np.array([1.0])
        if n_ys == 3:
            w = 1 / (2 - 2 ** (1 / 3)) * np.ones(3)
            w[1] = 1 - 2 * w[0]
        if n_ys == 5:
            w = 1 / (4 - 4 ** (1 / 3)) * np.ones(5)
            w[2] = 1 - 4 * w[0]
        return w
