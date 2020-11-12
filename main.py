"""
Liquid argon:
_____________
mass = 6.69E-26 kilograms
distance = sigma = 0.341E-9 meters
energy = epsilon = 1.65E-21 joules
temperature = epsilon / k_B = 119.8 kelvin
tau = sigma * sqrt(mass / epsilon) = 2.17E-12 seconds
velocity = sigma / tau = 1.57E2 m/s
force = epsilon/sigma = 4.85E-12 newtons
pressure = epsilon / (sigma ^ 3) = 4.2E7 pascal = 4.2E2 atmospheres

"""


from cProfile import run
from json import load
from math import pi
from datetime import datetime
from os import getcwd
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from constants import DATA_DIR, IS_LOGGED
from dynamic_parameters import SystemDynamicParameters
from external_parameters import ExternalParameters
from helpers import get_empty_float_scalars, get_empty_int_scalars, get_empty_vectors, sign
from log_config import debug_info, logger, logger_wraps
from modeling_parameters import ModelingParameters
from numba_procedures import lf_cycle, update_list_cycle
from potential_parameters import PotentialParameters
from static_parameters import SystemStaticParameters


class MolecularDynamics:

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
        self.verlet_lists = {
            'marker_1': get_empty_int_scalars(self.static.particles_number),
            'marker_2': get_empty_int_scalars(self.static.particles_number),
            'list': get_empty_int_scalars(100 * self.static.particles_number),
        }

    @logger_wraps()
    def system_dynamics(
            self,
            stage_id: int,
            thermostat_type: str,
            **kwargs,
    ):
        if thermostat_type == 'velocity_scaling':
            if stage_id == 1:
                return self.velocity_scaling_1()
            if stage_id == 2:
                self.velocity_scaling_2(**kwargs)
        elif thermostat_type == 'nose_hoover':
            raise ValueError('`nose_hoover` does not exist.')

    @logger_wraps()
    def velocity_scaling_1(self):
        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
        )
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )

        # if temperature == 0:
        #     temperature = self.model.initial_temperature
        # lmbd = (self.external.temperature / temperature) ** 0.5

        self.model.initial_temperature += (
                sign(self.external.temperature - self.model.initial_temperature)
                * self.external.heating_velocity
                * self.model.time_step
        )
        if temperature == 0:
            temperature = self.model.initial_temperature
        lmbd = (self.model.initial_temperature / temperature) ** 0.5

        debug_info(f'Kinetic Energy: {system_kinetic_energy};')
        debug_info(f'Temperature before velocity_scaling_1: {temperature};')
        debug_info(f'Initial Temperature: {self.model.initial_temperature};')
        debug_info(f'Lambda: {lmbd};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            vel_coefficient=lmbd,
        )
        debug_info(f'Temperature after velocity_scaling_1: {self.dynamic.temperature()};')
        return system_kinetic_energy, temperature

    @logger_wraps()
    def velocity_scaling_2(
            self,
            potential_energy: float,
            system_kinetic_energy: float,
            pressure: float,
    ):
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        debug_info(f'Pressure: {pressure};')
        debug_info(f'Temperature before velocity_scaling_2: {temperature};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
        total_energy = system_kinetic_energy + potential_energy
        debug_info(f'Temperature after velocity_scaling_2: {self.dynamic.temperature()};')
        return pressure, total_energy

    def load_forces(
            self,
            potential_table: np.ndarray,
    ):
        debug_info(f"Entering 'load_forces(potential_table)'")
        potential_energies = get_empty_float_scalars(self.static.particles_number)
        self.dynamic.accelerations = get_empty_vectors(self.static.particles_number)
        if self.potential.update_test:
            debug_info(f'update_test = True')
            self.update_list()
            self.dynamic.displacements = get_empty_vectors(self.static.particles_number)
            self.potential.update_test = False

        virial = lf_cycle(
            particles_number=self.static.particles_number,
            verlet_list=self.verlet_lists['list'],
            marker_1=self.verlet_lists['marker_1'],
            marker_2=self.verlet_lists['marker_2'],
            r_cut=self.potential.r_cut,
            potential_table=potential_table,
            potential_energies=potential_energies,
            positions=self.dynamic.positions,
            accelerations=self.dynamic.accelerations,
            cell_dimensions=self.static.cell_dimensions,
        )
        acc_mag = (self.dynamic.accelerations ** 2).sum(axis=1) ** 0.2
        debug_info(f'Mean and max acceleration: {acc_mag.mean()}, {acc_mag.max()}')
        potential_energy = potential_energies.sum()
        self.dynamic.displacements += (
                self.dynamic.velocities * self.model.time_step
                + self.dynamic.accelerations * self.model.time_step * self.model.time_step / 2.0
        )
        self.load_move_test()
        debug_info(f'Potential energy: {potential_energy};')
        debug_info(f"Exiting 'load_forces(potential_table)'")
        return potential_energy, virial

    # @staticmethod
    # @numba.jit(nopython=True)
    # def lf_cycle(
    #         particles_number,
    #         verlet_list,
    #         marker_1,
    #         marker_2,
    #         r_cut,
    #         potential_table,
    #         potential_energies,
    #         positions,
    #         accelerations,
    #         cell_dimensions,
    # ):
    #     virial = 0
    #     for i in range(particles_number - 1):
    #         # for j in range(i + 1, particles_number):
    #         for k in range(
    #                 marker_1[i],
    #                 marker_2[i] + 1,
    #         ):
    #             j = verlet_list[k]
    #             radius_vector = get_radius_vector(
    #                 index_1=i,
    #                 index_2=j,
    #                 positions=positions,
    #                 cell_dimensions=cell_dimensions
    #             )
    #             distance_squared = (radius_vector ** 2).sum()
    #             if distance_squared < r_cut * r_cut:
    #                 # TODO обеспечить защиту от перекрытий
    #                 # При расстояниях меньших 0.5 получаем отрицательный индекс
    #                 # А значит неправильные потенциал и ускорение.
    #                 table_row = int((distance_squared ** 0.5 - 0.5) / 0.0001)
    #                 # table_row = int((distance_squared ** 0.5) / 0.0001)
    #                 potential_ij = potential_table[table_row - 1, 0]
    #                 force_ij = potential_table[table_row - 1, 1]  # / (distance_squared ** 0.5)
    #                 potential_energies[i] += potential_ij / 2.0
    #                 potential_energies[j] += potential_ij / 2.0
    #                 virial += force_ij * distance_squared
    #                 acceleration_ij = force_ij * radius_vector
    #                 accelerations[i] += acceleration_ij
    #                 accelerations[j] -= acceleration_ij
    #                 assert table_row >= 1
    #     return virial

    @logger_wraps()
    def update_list(self):
        self.verlet_lists = {
            'marker_1': get_empty_int_scalars(self.static.particles_number),
            'marker_2': get_empty_int_scalars(self.static.particles_number),
            'list': get_empty_int_scalars(100 * self.static.particles_number),
        }
        advances = get_empty_int_scalars(self.static.particles_number)
        update_list_cycle(
            rng=self.potential.r_cut + self.potential.skin,
            advances=advances,
            particles_number=self.static.particles_number,
            positions=self.dynamic.positions,
            cell_dimensions=self.static.cell_dimensions,
            marker_1=self.verlet_lists['marker_1'],
            marker_2=self.verlet_lists['marker_2'],
            verlet_list=self.verlet_lists['list'],
        )

    # @staticmethod
    # @numba.jit(nopython=True)
    # def _update_list(
    #         rng: float,
    #         advances: np.ndarray,
    #         particles_number: int,
    #         positions: np.ndarray,
    #         cell_dimensions: np.ndarray,
    #         marker_1: np.ndarray,
    #         marker_2: np.ndarray,
    #         verlet_list: np.ndarray,
    # ):
    #     k = 1
    #     for i in range(particles_number - 1):
    #         for j in range(i + 1, particles_number):
    #             radius_vector = get_radius_vector(
    #                 index_1=i,
    #                 index_2=j,
    #                 positions=positions,
    #                 cell_dimensions=cell_dimensions
    #             )
    #             distance_squared = (radius_vector ** 2).sum()
    #             if distance_squared < rng * rng:
    #                 advances[j] = 1
    #             else:
    #                 advances[j] = 0
    #
    #         marker_1[i] = k
    #         for j in range(i + 1, particles_number):
    #             verlet_list[k] = j
    #             k = k + advances[j]
    #         marker_2[i] = k - 1

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

    def save_config(self, file_name: str = None):
        file_name = join(DATA_DIR, file_name or 'system_config_149.txt')
        with open(file_name, mode='w', encoding='utf-8') as file:
            lines = [
                '\n'.join(self.static.cell_dimensions.astype(str)),
                str(self.static.particles_number),
                str(self.model.time),
                *[
                    f'{pos[0]} {pos[1]} {pos[2]}'
                    for i, pos in enumerate(self.dynamic.positions)
                ],
                *[
                    f'{vel[0]} {vel[1]} {vel[2]}'
                    for i, vel in enumerate(self.dynamic.velocities)
                ],
                *[
                    f'{acc[0]} {acc[1]} {acc[2]}'
                    for i, acc in enumerate(self.dynamic.accelerations)
                ],
            ]
            file.write('\n'.join(lines))

    @logger_wraps()
    def load_save_config(self, file_name: str = None):
        file_name = join(DATA_DIR, file_name or 'system_config_149.txt')
        with open(file_name, mode='r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            self.static.cell_dimensions = np.array(lines[:3], dtype=np.float)
            self.static.particles_number = int(lines[3])
            self.model.time = float(lines[4])
            self.dynamic.positions = get_empty_vectors(self.static.particles_number)
            self.dynamic.velocities = get_empty_vectors(self.static.particles_number)
            self.dynamic.accelerations = get_empty_vectors(self.static.particles_number)
            for i in range(self.static.particles_number):
                self.dynamic.positions[i] = np.array(
                    lines[5 + i].split(),
                    dtype=np.float,
                )
                self.dynamic.velocities[i] = np.array(
                    lines[5 + i + self.static.particles_number].split(),
                    dtype=np.float,
                )
                self.dynamic.velocities[i] = np.array(
                    lines[5 + i + 2 * self.static.particles_number].split(),
                    dtype=np.float,
                )

    @staticmethod
    def save_system_parameters(
            system_parameters: dict,
            step: int,
            potential_energy: float,
            temperature: float,
            pressure: float,
            system_kinetic_energy: float,
    ):
        system_parameters['temperature'][step - 1] = temperature
        system_parameters['pressure'][step - 1] = pressure
        system_parameters['kinetic_energy'][step - 1] = system_kinetic_energy
        system_parameters['potential_energy'][step - 1] = potential_energy
        system_parameters['total_energy'][step - 1] = system_kinetic_energy + potential_energy

    def load_lammps_trajectory(self):
        lines = [
            'ITEM: TIMESTEP',
            str(self.model.time),
            'ITEM: NUMBER OF ATOMS',
            str(self.static.particles_number),
            'ITEM: BOX BOUNDS pp pp pp',
            *(
                f'{-dim / 2} {dim / 2}'
                for dim in self.static.cell_dimensions
            ),
            'ITEM: ATOMS id type x y z',
            *(
                f'{i + 1} 0 {pos[0]} {pos[1]} {pos[2]}'
                for i, pos in enumerate(self.dynamic.positions)
            ),
            '\n',
        ]
        return '\n'.join(lines)

    def md_time_step(
            self,
            lammps_trajectories: list,
            potential_table: np.ndarray,
            step: int,
            system_parameters: dict,
    ):
        system_kinetic_energy, temperature = self.system_dynamics(
            stage_id=1,
            thermostat_type='velocity_scaling',
        )
        system_center_1 = self.dynamic.system_center
        debug_info(f'System center after system_dynamics_1: {system_center_1}')
        system_center_2 = self.dynamic.system_center
        if IS_LOGGED and any(system_center_2 > 1e-10):
            logger.warning(f'Drift of system center!')
        debug_info(f'System center after boundary_conditions: {system_center_2}')

        potential_energy, virial = self.load_forces(
            potential_table=potential_table,
        )
        pressure = self.dynamic.get_pressure(
            temperature=temperature,
            virial=virial,
        )
        parameters = {
            'system_kinetic_energy': system_kinetic_energy,
            'potential_energy': potential_energy,
            'pressure': pressure,
        }

        self.system_dynamics(
            stage_id=2,
            thermostat_type='velocity_scaling',
            **parameters,
        )
        self.save_system_parameters(
            system_parameters=system_parameters,
            step=step,
            temperature=temperature,
            **parameters
        )
        if step % 20 == 0:
            lammps_trajectories.append(
                self.load_lammps_trajectory()
            )
        print(
            f'Step: {step}/{self.model.iterations_numbers};',
            f'\tTime = {self.model.time:.3f};',
            f'\tT = {temperature:.5f};',
            f'\tP = {pressure:.5f};\n',
            sep='\n',
        )
        debug_info(f'End of step {step}.\n')
        if step % 1000 == 0:
            _start = datetime.now()
            with open(
                    join(DATA_DIR, 'system_config.txt'),
                    mode='a',
                    encoding='utf-8'
            ) as file:
                file.write('\n'.join(lammps_trajectories))
            print(
                f'LAMMPS trajectories for last 1000 steps are saved. '
                f'Time of saving: {datetime.now() - _start}'
            )
            lammps_trajectories = []

    @logger_wraps()
    def run_md(self):
        start = datetime.now()
        potential_table = self.potential.potential_table
        lammps_trajectories = []
        system_parameters = {
            'temperature': get_empty_float_scalars(self.model.iterations_numbers),
            'pressure': get_empty_float_scalars(self.model.iterations_numbers),
            'kinetic_energy': get_empty_float_scalars(self.model.iterations_numbers),
            'potential_energy': get_empty_float_scalars(self.model.iterations_numbers),
            'total_energy': get_empty_float_scalars(self.model.iterations_numbers),
        }
        for step in range(1, self.model.iterations_numbers + 1):
            self.model.time += self.model.time_step
            debug_info(f'Step: {step}; Time: {self.model.time:.3f};')
            self.md_time_step(
                lammps_trajectories=lammps_trajectories,
                potential_table=potential_table,
                step=step,
                system_parameters=system_parameters,
            )

        print(f'Calculation completed. Time of calculation: {datetime.now() - start}')

        _start = datetime.now()
        with open(
                join(DATA_DIR, 'system_config.txt'),
                mode='a',
                encoding='utf-8'
        ) as file:
            file.write('\n'.join(lammps_trajectories))
        print(
            f'LAMMPS trajectories for last {self.model.iterations_numbers % 1000} steps are saved. '
            f'Time of saving: {datetime.now() - _start}'
        )

        start = datetime.now()
        DataFrame(system_parameters).to_csv(
            join(DATA_DIR, 'system_parameters.csv'),
            sep=';',
            index=False,
        )

        print(f'System parameters are saved. Time of saving: {datetime.now() - start}')

    def run_rdf(self, steps_number: int = 1000, file_name: str = None):
        file_name = join(DATA_DIR, file_name or 'system_config_149.txt')
        layer_thickness = 0.01
        begin_step = 1
        end_step = steps_number
        rdf = get_empty_float_scalars(10 * self.static.particles_number)
        with open(file_name, mode='r', encoding='utf8') as file:
            lines = file.readlines()
            for step in range(begin_step, end_step + 1):
                for i in range(self.static.particles_number):
                    self.dynamic.positions[i] = np.array(
                        lines[9 + i].split()[2:],
                        dtype=np.float,
                    )
                distances = self.dynamic.interparticle_distances
                layer_numbers = (distances / layer_thickness).astype(np.int)
                max_layer_number = layer_numbers.max()
                particles_in_layer = get_empty_int_scalars(max_layer_number + 1)
                layers, particles = np.unique(layer_numbers, return_counts=True)
                for i, layer in enumerate(layers):
                    particles_in_layer[layer] = particles[i]
                radiuses = layer_thickness * (np.arange(max_layer_number + 1) + 0.5)
                rdf[:max_layer_number + 1] += (
                        2.0 * self.static.get_cell_volume()
                        / (4.0 * pi * radiuses * radiuses
                           * self.static.particles_number * self.static.particles_number)
                        * particles_in_layer / layer_thickness
                )
                print(f'Step: {step}/{end_step};')
        with open('rdf_file.txt', mode='w', encoding='utf8') as file:
            rdf = rdf[:np.nonzero(rdf)[0][-1]]
            radiuses = layer_thickness * (np.arange(rdf.size) + 0.5)
            rdf = rdf / (end_step - begin_step + 1)
            for i, radius in enumerate(radiuses):
                if radius <= self.static.cell_dimensions[0] / 2.0:
                    file.write(f'{radius} {rdf[i]}\n')
        print('Calculation completed.')

    @staticmethod
    def plot_rdf(file_name: str):
        rdf_data = []
        with open('rdf_file.txt', mode='r', encoding='utf8') as file:
            for line in file:
                rdf_data.append(np.array(line.rstrip().split()).astype(np.float))

        rdf_data = np.array(rdf_data).transpose()
        plt.plot(*rdf_data)
        plt.xlabel(r'r/$\sigma$')
        plt.ylabel('g(r)')
        plt.ylim(bottom=0, top=100)
        plt.savefig(file_name)


def main(config_filename: str = None):
    _config_filename = join(
        getcwd(),
        config_filename or 'config.json'
    )
    with open(_config_filename, encoding='utf8') as file:
        config_parameters = load(file)

    static = SystemStaticParameters(**config_parameters['static_parameters'])
    external = ExternalParameters(**config_parameters['external_parameters'])
    model = ModelingParameters(**config_parameters['modeling_parameters'])
    dynamic = SystemDynamicParameters(
        static=static,
        # temperature=model.initial_temperature,
    )
    potential = PotentialParameters(**config_parameters['potential_parameters'])
    md_sample = MolecularDynamics(
        static=static,
        dynamic=dynamic,
        external=external,
        model=model,
        potential=potential,
    )
    # run(
    #     'md_sample.run_md()',
    #     sort=2,
    # )
    md_sample.run_md()


if __name__ == '__main__':
    np.set_printoptions(threshold=5000)
    main()
