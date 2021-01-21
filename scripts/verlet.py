import numpy as np

from scripts.constants import TEMPERATURE_MINIMUM
from scripts.dynamic_parameters import SystemDynamicParameters
from scripts.external_parameters import ExternalParameters
from scripts.helpers import get_empty_float_scalars, get_empty_int_scalars, get_empty_vectors, sign
from scripts.log_config import log_debug_info, logger_wraps
from scripts.modeling_parameters import ModelingParameters
from scripts.numba_procedures import lf_cycle, update_list_cycle
from scripts.potential_parameters import PotentialParameters
from scripts.static_parameters import SystemStaticParameters


class Verlet:

    def __init__(
            self,
            static: SystemStaticParameters,
            dynamic: SystemDynamicParameters,
            external: ExternalParameters,
            model: ModelingParameters,
            potential: PotentialParameters,
    ):
        self.potential = potential
        self.static = static
        self.dynamic = dynamic
        self.external = external
        self.model = model
        self.neighbours_lists = {
            'all_neighbours': get_empty_int_scalars(self.static.particles_number),
            'first_neighbours': get_empty_int_scalars(self.static.particles_number),
            'last_neighbours': get_empty_int_scalars(100 * self.static.particles_number),
        }
        self.npt_factor = 0.0
        self.nvt_factors = np.zeros(self.external.thermostat_parameters.size, dtype=np.float)
        self.xis = np.zeros(self.external.thermostat_parameters.size, dtype=np.float)
        self.s_f = 0.0
        self.epsilon = np.log(self.static.get_cell_volume() / 3)

    @logger_wraps()
    def system_dynamics(
            self,
            stage_id: int,
            environment_type: str,
            **kwargs,
    ):
        if stage_id == 1:
            return self.stage_1(algorithm_name=environment_type)
        if stage_id == 2:
            if environment_type == 'velocity_scaling':
                return self.velocity_scaling_2(**kwargs)
            elif environment_type == 'nose_hoover':
                return self.nose_hoover_2(**kwargs)

    @logger_wraps()
    def stage_1(self, algorithm_name: str):
        acc_coefficient = 0
        if algorithm_name == 'nose_hoover':
            acc_coefficient = self.nvt_factors[0] + self.npt_factor

        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
            acc_coefficient=acc_coefficient,
        )
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        self.model.initial_temperature += (
                sign(self.external.temperature - self.model.initial_temperature)
                * self.external.heating_velocity
                * self.model.time_step
        )
        if self.model.initial_temperature < TEMPERATURE_MINIMUM:
            self.model.initial_temperature = TEMPERATURE_MINIMUM
        if temperature <= 0:
            temperature = self.model.initial_temperature

        if algorithm_name == 'velocity_scaling':
            self.nvt_factors[0] = (self.model.initial_temperature / temperature) ** 0.5
        elif algorithm_name == 'berendsen':
            self.nvt_factors[0] = (1 + self.model.time_step / self.external.thermostat_parameters[0]
                                   * (self.model.initial_temperature / temperature - 1)) ** 0.5
        elif algorithm_name == 'nose_hoover':
            lambda_diff = ((2.0 * system_kinetic_energy
                            - 3.0 * self.static.particles_number * self.model.initial_temperature)
                           * self.model.time_step / self.external.thermostat_parameters[0])
            self.s_f += self.nvt_factors[0] * self.model.time_step + lambda_diff * self.model.time_step
            self.nvt_factors[0] += lambda_diff / 2.0

        log_debug_info(f'Kinetic Energy: {system_kinetic_energy};')
        log_debug_info(f'Temperature before {algorithm_name}_1: {temperature};')
        log_debug_info(f'Initial Temperature: {self.model.initial_temperature};')
        log_debug_info(f'NVT-factor: {self.nvt_factors[0]};')

        vel_coefficient = 1
        if algorithm_name == 'velocity_scaling':
            vel_coefficient = self.nvt_factors[0]
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            vel_coefficient=vel_coefficient,
        )
        log_debug_info(f'Temperature after {algorithm_name}_1: {self.dynamic.temperature()};')
        return system_kinetic_energy, temperature

    @logger_wraps()
    def velocity_scaling_2(
            self,
            potential_energy: float,
            system_kinetic_energy: float,
    ):
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
        )
        log_debug_info(f'Pressure before velocity_scaling_2: {pressure};')
        log_debug_info(f'Temperature before velocity_scaling_2: {temperature};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
        total_energy = system_kinetic_energy + potential_energy
        log_debug_info(f'Temperature after velocity_scaling_2: {self.dynamic.temperature()};')
        return pressure, total_energy

    @logger_wraps()
    def nose_hoover_2(
            self,
            potential_energy: float,
            system_kinetic_energy: float,
    ):
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        cell_volume = self.static.get_cell_volume()
        density = self.static.get_density()
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            cell_volume=cell_volume,
        )
        log_debug_info(f'Cell Volume before nose_hoover_2: {cell_volume};')
        log_debug_info(f'Density before nose_hoover_2: {density};')
        log_debug_info(f'Temperature before nose_hoover_2: {temperature};')
        log_debug_info(f'Pressure before nose_hoover_2: {pressure};')
        self.npt_factor += (
                (pressure - self.external.pressure)
                * self.model.time_step
                / self.external.barostat_parameter / density
        )
        log_debug_info(f'NPT-factor: {self.npt_factor};')
        self.static.cell_dimensions *= 1.0 + self.npt_factor * self.model.time_step

        cell_volume = self.static.get_cell_volume()
        density = self.static.get_density()
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            cell_volume=cell_volume,
        )
        log_debug_info(f'New cell dimensions: {self.static.cell_dimensions};')
        log_debug_info(f'New cell volume: {cell_volume};')
        log_debug_info(f'New density: {density};')
        log_debug_info(f'Pressure at new volume: {pressure};')

        # TODO incorrect algorithm
        error = 1e-5
        half_time = self.model.time_step / 2.0
        time_ratio = self.model.time_step / self.external.thermostat_parameters[0]
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        nvt_factor_new = self.nvt_factors[0]
        velocities_new = self.dynamic.velocities
        is_ready = False
        for i in range(50):
            log_debug_info(f'i: {i}')
            if is_ready:
                break
            nvt_factor_old = nvt_factor_new
            ds = 0
            velocities_old = velocities_new
            b = (
                    -half_time
                    * (self.dynamic.accelerations - nvt_factor_old * velocities_new)
                    - (self.dynamic.velocities - velocities_old)
            )
            ds += (velocities_old * b).sum() * time_ratio
            dd = 1 - nvt_factor_old * half_time
            ds -= dd * (
                    (
                            3.0 * self.static.particles_number
                            * self.model.initial_temperature
                            - 2.0 * system_kinetic_energy
                    )
                    * time_ratio / 2.0
                    - (self.nvt_factors[0] - nvt_factor_old)
            )
            ds = ds / (-self.model.time_step * system_kinetic_energy * time_ratio + dd)
            velocities_new += (b + half_time * velocities_old * ds) / dd
            system_kinetic_energy = (velocities_new * velocities_new).sum() / 2.0
            nvt_factor_new = nvt_factor_old + ds

            is_ready = (
                    (abs((velocities_new - velocities_old) / velocities_new) <= error).all()
                    and (abs((nvt_factor_new - nvt_factor_old) / nvt_factor_new) <= error).all()
            )

        self.dynamic.velocities = velocities_new
        self.nvt_factors[0] = nvt_factor_new
        total_energy = (
                system_kinetic_energy + potential_energy
                + self.nvt_factors[0] * self.nvt_factors[0] * self.external.thermostat_parameters[0] / 2.0
                + 3.0 * self.static.particles_number * self.model.initial_temperature * self.s_f
        )

        cell_volume = self.static.get_cell_volume()
        temperature = self.dynamic.temperature()
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            cell_volume=cell_volume,
        )
        log_debug_info(f'Temperature after nose_hoover_2: {temperature};')
        log_debug_info(f'Pressure after nose_hoover_2: {pressure};')
        return pressure, total_energy

    def load_forces(
            self,
            potential_table: np.ndarray,
    ):
        log_debug_info(f"Entering 'load_forces(potential_table)'")
        potential_energies = get_empty_float_scalars(self.static.particles_number)
        self.dynamic.accelerations = get_empty_vectors(self.static.particles_number)
        if self.potential.update_test:
            log_debug_info(f'update_test = True')
            self.update_list()
            self.dynamic.displacements = get_empty_vectors(self.static.particles_number)
            self.potential.update_test = False

        self.dynamic.virial = lf_cycle(
            particles_number=self.static.particles_number,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
            r_cut=self.potential.r_cut,
            potential_table=potential_table,
            potential_energies=potential_energies,
            positions=self.dynamic.positions,
            accelerations=self.dynamic.accelerations,
            cell_dimensions=self.static.cell_dimensions,
        )
        acc_mag = (self.dynamic.accelerations ** 2).sum(axis=1) ** 0.5
        log_debug_info(f'Mean and max acceleration: {acc_mag.mean()}, {acc_mag.max()}')
        potential_energy = potential_energies.sum()
        self.dynamic.displacements += (
                self.dynamic.velocities * self.model.time_step
                + self.dynamic.accelerations * self.model.time_step * self.model.time_step / 2.0
        )
        self.load_move_test()
        log_debug_info(f'Potential energy: {potential_energy};')
        log_debug_info(f"Exiting 'load_forces(potential_table)'")
        self.dynamic.potential_energy = potential_energy

    @logger_wraps()
    def update_list(self):
        self.neighbours_lists = {
            'first_neighbours': get_empty_int_scalars(self.static.particles_number),
            'last_neighbours': get_empty_int_scalars(self.static.particles_number),
            'all_neighbours': get_empty_int_scalars(100 * self.static.particles_number),
        }
        advances = get_empty_int_scalars(self.static.particles_number)
        update_list_cycle(
            rng=self.potential.r_cut + self.potential.skin,
            advances=advances,
            particles_number=self.static.particles_number,
            positions=self.dynamic.positions,
            cell_dimensions=self.static.cell_dimensions,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
        )

    @logger_wraps()
    def load_move_test(self):
        ds_1, ds_2 = 0.0, 0.0
        for i in range(self.static.particles_number):
            ds = (self.dynamic.displacements[i] ** 2).sum() ** 0.5
            if ds >= ds_1:
                ds_2 = ds_1
                ds_1 = ds
            elif ds >= ds_2:
                ds_2 = ds
        self.potential.update_test = (ds_1 + ds_2) > self.potential.skin

    def mtk_integrate(self):
        fact_3 = 6
        fact_5 = fact_3 * 20
        fact_7 = fact_5 * 42
        fact_9 = fact_7 * 72
        self.mtk_npt_liouville()
        self.dynamic.get_next_velocities(time_step=self.model.time_step)
        vlogv_dt2 = self.model.time_step ** 2 * self.npt_factor
        aa = np.exp(vlogv_dt2)
        aa_2 = aa * aa
        arg_2 = vlogv_dt2 * vlogv_dt2
        # Разложение -sin(x) / x
        poly = 1.0 + arg_2 / fact_3 + arg_2 ** 2 / fact_5 + arg_2 ** 3 / fact_7 + arg_2 ** 4 / fact_9
        bb = aa * poly * self.model.time_step
        self.dynamic.positions = self.dynamic.positions * aa_2 + self.dynamic.velocities * bb
        self.epsilon += self.npt_factor * self.model.time_step
        self.load_forces(potential_table=self.potential.potential_table)
        self.dynamic.get_next_velocities(time_step=self.model.time_step)
        self.mtk_npt_liouville()

    def get_g_npt(self, system_kinetic_energy: float):
        # TODO
        pint = self.dynamic.get_pressure()
        return (
                       (1 + 1 / self.static.particles_number) * 2 * system_kinetic_energy
                       + 3.0 * self.static.get_cell_volume() * (pint - self.external.pressure)
               ) / self.external.barostat_parameter

    def get_g_nvt_0(self, system_kinetic_energy: float):
        return (
                       2 * system_kinetic_energy
                       + self.external.barostat_parameter * self.npt_factor * self.npt_factor
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

    def do_liouville_nvt_last(
            self,
            thermostats_number: int,
            g_nvt: np.ndarray,
            dt: float,
    ):
        self.nvt_factors[thermostats_number - 1] += g_nvt[thermostats_number - 1] * dt / 4

    def do_liouville_nvt_not_last(
            self,
            thermostats_number: int,
            g_nvt: np.ndarray,
            dt: float,
            is_right: bool,
    ):
        for j in range(thermostats_number - 1):
            k = thermostats_number - 2 - j if is_right else j
            aa = np.exp(self.nvt_factors[k + 1] * dt / 8)
            self.nvt_factors[k] *= aa
            self.nvt_factors[k] += g_nvt[k] * dt / 4
            self.nvt_factors[k] *= aa
            if not is_right:
                g_nvt[j + 1] = self.get_g_nvt_k(k=j + 1)
        return g_nvt

    def do_liouville_npt(
            self,
            g_npt: float,
            dt: float,
    ):
        aa = np.exp(-dt / 8 * self.nvt_factors[0])
        self.npt_factor *= aa
        self.npt_factor += g_npt * dt / 4
        self.npt_factor *= aa

    def do_liouville_velocities(
            self,
            dt: float,
            system_kinetic_energy: float,
            scale: float,
    ):
        aa = np.exp(
            -dt / 2
            * (
                    self.nvt_factors[0]
                    + (1 + 1 / self.static.particles_number) * self.npt_factor
            )
        )
        scale *= aa
        system_kinetic_energy *= aa * aa
        g_npt = self.get_g_npt(system_kinetic_energy=system_kinetic_energy)
        return scale, system_kinetic_energy, g_npt

    def do_liouville_xis(
            self,
            dt: float,
            thermostats_number: int,
    ):
        for j in range(thermostats_number):
            self.xis[j] += self.nvt_factors[j] * dt / 2

    def mtk_npt_liouville(self):
        # TODO check operators order and g_nvt calculation
        scale, n_c, n_ys = 1.0, 1, 3
        w = np.array([1 / (2 - 2 ** (1 / 3)), 1 - 2 / (2 - 2 ** (1 / 3)), 1 / (2 - 2 ** (1 / 3))])
        thermostats_number = self.external.thermostat_parameters.size
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        g_nvt = np.zeros(thermostats_number, dtype=np.float)
        g_nvt[0] = self.get_g_nvt_0(system_kinetic_energy=system_kinetic_energy)

        for k in range(1, g_nvt.size):
            g_nvt[k] = self.get_g_nvt_k(k=k)

        g_npt = self.get_g_npt(system_kinetic_energy=system_kinetic_energy)

        for _ in range(n_c):
            for i in range(n_ys):
                dt = w[i] * self.model.time_step / n_c
                self.do_liouville_nvt_last(
                    thermostats_number=thermostats_number,
                    g_nvt=g_nvt,
                    dt=dt,
                )
                g_nvt = self.do_liouville_nvt_not_last(
                    thermostats_number=thermostats_number,
                    g_nvt=g_nvt,
                    dt=dt,
                    is_right=True,
                )
                self.do_liouville_npt(
                    g_npt=g_npt,
                    dt=dt,
                )
                scale, system_kinetic_energy, g_npt = self.do_liouville_velocities(
                    dt=dt,
                    system_kinetic_energy=system_kinetic_energy,
                    scale=scale,
                )
                self.do_liouville_xis(
                    dt=dt,
                    thermostats_number=thermostats_number,
                )
                self.do_liouville_npt(
                    g_npt=g_npt,
                    dt=dt,
                )
                g_nvt[0] = self.get_g_nvt_0(system_kinetic_energy=system_kinetic_energy)
                g_nvt = self.do_liouville_nvt_not_last(
                    thermostats_number=thermostats_number,
                    g_nvt=g_nvt,
                    dt=dt,
                    is_right=False,
                )
                self.do_liouville_nvt_last(
                    thermostats_number=thermostats_number,
                    g_nvt=g_nvt,
                    dt=dt,
                )

        self.dynamic.velocities *= scale
