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


# from cProfile import run
from math import pi
import logging
from os import getcwd
from os.path import join
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import numba
from pandas import DataFrame

from dynamic_parameters import SystemDynamicParameters
from external_parameters import ExternalParameters
from helpers import get_empty_float_scalars, get_empty_int_scalars, get_empty_vectors
from modeling_parameters import ModelingParameters
from potential_parameters import PotentialParameters
from static_parameters import SystemStaticParameters


DATA_DIR = join(getcwd(), 'data')

IS_LOGGING = True

if IS_LOGGING:
    logging.basicConfig(
        level=logging.DEBUG,
        filename=join(DATA_DIR, 'log.txt'),
        format='%(asctime)s - %(message)s'
    )


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
            if stage_id == 1:
                self.nose_hoover_1()
            if stage_id == 2:
                self.nose_hoover_2(**kwargs)

    def velocity_scaling_1(self):
        if IS_LOGGING:
            logging.debug('velocity_scaling_1 - In;')

        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
        )
        if IS_LOGGING:
            logging.debug('got next positions;')
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        temperature = self.dynamic.temperature(
            system_kinetic_energy=system_kinetic_energy,
        )
        self.model.initial_temperature += (
                (1 if (self.external.temperature - self.model.initial_temperature) >= 0 else -1)
                * self.external.heating_velocity
                * self.model.time_step
        )
        if temperature == 0:
            temperature = self.model.initial_temperature
        if IS_LOGGING:
            logging.debug(f'Kinetic energy: {system_kinetic_energy}, temperature: {temperature}')
            logging.debug(f'Initial temperature: {self.model.initial_temperature};')
            logging.debug(f'Lambda: {np.sqrt(self.model.initial_temperature / temperature)};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            vel_coefficient=np.sqrt(self.model.initial_temperature / temperature),
        )
        if IS_LOGGING:
            logging.debug(f'Temperature after scaling: {self.dynamic.temperature()};')
            logging.debug('velocity_scaling_1 - Out;\n')
        return system_kinetic_energy, temperature

    def velocity_scaling_2(
            self,
            temperature: float = None,
            virial: float = None,
            system_kinetic_energy: float = None,
            potential_energy: float = None,
            pressure: float = None,
    ):
        if IS_LOGGING:
            logging.debug('velocity_scaling_2 - In;')
        _system_kinetic_energy = system_kinetic_energy or self.dynamic.system_kinetic_energy
        _temperature = temperature or self.dynamic.temperature(
            system_kinetic_energy=_system_kinetic_energy,
        )
        if not all((potential_energy, virial)):
            _potential_energy, _virial = self.load_forces(
                potential_table=self.potential.potential_table,
            )
        else:
            _potential_energy, _virial = potential_energy, virial
        _pressure = pressure or self.dynamic.get_pressure(
            temperature=_temperature,
            virial=_virial,
        )
        if IS_LOGGING:
            logging.debug(f'Pressure: {_pressure};')
            logging.debug(f'Temperature: {_temperature};')
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
        total_energy = _system_kinetic_energy + _potential_energy
        if IS_LOGGING:
            logging.debug(f'Temperature after scaling: {self.dynamic.temperature()};')
            logging.debug('velocity_scaling_2 - Out;\n')
        return _pressure, total_energy

    def nose_hoover_1(self):
        lmbd = self.external.parameters['lambda']
        xi = self.external.parameters['xi']
        nose_hoover_thermostat_parameter = self.external.parameters['Q_T']
        s_f = self.external.parameters['S_f']
        self.dynamic.get_next_positions(
            time_step=self.model.time_step,
            acc_coefficient=(lmbd + xi),
        )
        system_kinetic_energy = self.dynamic.system_kinetic_energy
        temperature = self.dynamic.temperature
        if temperature == 0:
            temperature = self.model.initial_temperature
        temperature += (
                (1 if (self.external.temperature - temperature) >= 0 else -1)
                * self.external.heating_velocity
                * self.model.time_step
        )
        s_f += (
                lmbd * self.model.time_step
                + (
                        system_kinetic_energy
                        - 3.0 * self.static.particles_number * self.external.temperature
                ) * self.model.time_step * self.model.time_step / nose_hoover_thermostat_parameter
        )
        lmbd += (
                        system_kinetic_energy
                        - 3.0 * self.static.particles_number * self.external.temperature
                ) / nose_hoover_thermostat_parameter * self.model.time_step / 2.0
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            acc_coefficient=(lmbd + xi),
        )
        self.external.parameters['lambda'] = lmbd
        self.external.parameters['S_f'] = s_f

    def nose_hoover_2(
            self,
            temperature: float = None,
            virial: float = None,
            potential_energy: float = None,
    ):
        _temperature = temperature or self.dynamic.temperature
        xi = self.external.parameters['xi']
        nose_hoover_barostat_parameter = self.external.parameters['Q_B']
        if not all((potential_energy, virial)):
            _potential_energy, _virial = self.load_forces(
                potential_table=self.potential.potential_table,
            )
        else:
            _potential_energy, _virial = potential_energy, virial
        pressure = self.dynamic.get_pressure(
            temperature=_temperature,
            virial=_virial,
        )
        xi += (
                (pressure - self.external.pressure)
                * self.model.time_step
                / (self.static.get_density() * nose_hoover_barostat_parameter)
        )
        self.static.cell_dimensions *= 1.0 + xi * self.model.time_step
        pass  # TODO

    def load_forces(
            self,
            potential_table: np.ndarray,
    ):
        if IS_LOGGING:
            logging.debug('load_forces - In;')
        potential_energies = get_empty_float_scalars(self.static.particles_number)
        self.dynamic.accelerations = get_empty_vectors(self.static.particles_number)
        virial = 0
        distances = self.dynamic.interparticle_distances
        if self.potential.update_test:
            self.update_list(distances=distances)
            self.potential.update_test = False

        self.lf_cycle(
            particles_number=self.static.particles_number,
            verlet_list=self.verlet_lists['list'],
            marker_1=self.verlet_lists['marker_1'],
            marker_2=self.verlet_lists['marker_2'],
            distances=distances,
            r_cut=self.potential.r_cut,
            potential_table=potential_table,
            potential_energies=potential_energies,
            virial=virial,
            positions=self.dynamic.positions,
            accelerations=self.dynamic.accelerations,
        )
        potential_energy = potential_energies.sum()
        self.dynamic.displacements += (
                self.dynamic.velocities * self.model.time_step
                + self.dynamic.accelerations * self.model.time_step * self.model.time_step / 2.0
        )
        self.load_move_test()
        if IS_LOGGING:
            logging.debug(f'Potential energy: {potential_energy};')
            logging.debug('load_forces - Out;\n')
        return potential_energy, virial

    @staticmethod
    @numba.njit
    def lf_cycle(particles_number, verlet_list, marker_1, marker_2, distances, r_cut, potential_table,
                 potential_energies, virial, positions, accelerations):
        for i in range(particles_number - 1):
            for k in range(
                    marker_1[i],
                    marker_2[i] + 1,
            ):
                j = verlet_list[k]
                distance = distances[i][j]
                if distance < r_cut:
                    table_row = int((distance - 0.5) / 0.0001)
                    potential_ij = potential_table[table_row, 0]
                    force_ij = potential_table[table_row, 1]
                    potential_energies[j] += potential_ij
                    virial += force_ij * distance
                    acceleration_ij = (
                            force_ij
                            * (positions[i] - positions[j])
                            / distance
                    )
                    accelerations[i] += acceleration_ij
                    accelerations[j] -= acceleration_ij

    @staticmethod
    @numba.njit
    def lf_logic(i, j, distance, potential_table, potential_energies, virial, positions, accelerations):
        table_row = int((distance - 0.5) / 0.0001)
        potential_ij = potential_table[table_row, 0]
        force_ij = potential_table[table_row, 1]
        potential_energies[j] += potential_ij
        virial += force_ij * distance
        acceleration_ij = (
                force_ij
                * (positions[i] - positions[j])
                / distance
        )
        accelerations[i] += acceleration_ij
        accelerations[j] -= acceleration_ij

    def update_list(self, distances: np.ndarray):
        advances = (distances < (self.potential.r_cut + self.potential.skin)).astype(np.int)
        self.dynamic.displacements = self._update_list(
            advances=advances,
            particles_number=self.static.particles_number,
            marker_1=self.verlet_lists['marker_1'],
            marker_2=self.verlet_lists['marker_2'],
            verlet_list=self.verlet_lists['list'],
        )

        self.dynamic.displacements = get_empty_vectors(self.static.particles_number)

    @staticmethod
    @numba.jit(
        nopython=True,
    )
    def _update_list(
            advances: np.ndarray,
            particles_number: int,
            marker_1: np.ndarray,
            marker_2: np.ndarray,
            verlet_list: np.ndarray,
    ):
        k = 1
        for i in range(particles_number - 1):
            marker_1[i] = k
            for j in range(i + 1, particles_number):
                verlet_list[k] = j
                if advances[i][j]:
                    k += advances[i][j]
            marker_2[i] = k - 1

    def load_move_test(self):
        ds = (self.dynamic.displacements * self.dynamic.displacements).sum(axis=1) ** 0.5
        ds_1, ds_2 = 0.0, 0.0
        for i in range(self.static.particles_number):
            if ds[i] >= ds_1:
                ds_2 = ds_1
                ds_1 = ds[i]
            elif ds[i] >= ds_2:
                ds_2 = ds[i]
        self.potential.update_test = (ds_1 + ds_2) > self.potential.skin

    def save_config(self, file_name: str = None):
        file_name = join(DATA_DIR, file_name or 'system_config.txt')
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

    def load_save_config(self, file_name: str = None):
        file_name = join(DATA_DIR, file_name or 'system_config.txt')
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

    def save_system_parameters(
            self,
            system_parameters: dict,
            step: int,
            temperature: float = None,
            pressure: float = None,
            system_kinetic_energy: float = None,
            potential_energy: float = None,
            virial: float = None,
    ):
        _system_kinetic_energy = system_kinetic_energy or self.dynamic.system_kinetic_energy
        _temperature = temperature or self.dynamic.temperature
        if not all((potential_energy, virial)):
            _potential_energy, _virial = self.load_forces(
                potential_table=self.potential.potential_table,
            )
        else:
            _potential_energy, _virial = potential_energy, virial
        _pressure = pressure or self.dynamic.get_pressure(
            temperature=_temperature,
            virial=_virial,
        )
        system_parameters['temperature'][step - 1] = _temperature
        system_parameters['pressure'][step - 1] = _pressure
        system_parameters['kinetic_energy'][step - 1] = _system_kinetic_energy
        system_parameters['potential_energy'][step - 1] = _potential_energy
        system_parameters['total_energy'][step - 1] = _system_kinetic_energy + _potential_energy

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

    def run_md(self):
        start = datetime.now()
        if IS_LOGGING:
            logging.debug('Start.')
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
            if IS_LOGGING:
                logging.debug(f'Step: {step}; Time: {self.model.time:.3f};')
            system_kinetic_energy, temperature = self.system_dynamics(
                stage_id=1,
                thermostat_type='velocity_scaling',
            )
            self.dynamic.boundary_conditions()
            potential_energy, virial = self.load_forces(
                potential_table=potential_table,
            )
            pressure = self.dynamic.get_pressure(
                temperature=temperature,
                virial=virial,
            )
            parameters = {
                'virial': virial,
                'temperature': temperature,
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
                **parameters
            )
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
            if IS_LOGGING:
                logging.debug('End of iteration.\n')
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
            index=False
        )

        print(f'System parameters are saved. Time of saving: {datetime.now() - start}')

    def run_rdf(self, steps_number: int = 1000, file_name: str = None):
        file_name = join(DATA_DIR, file_name or 'system_config.txt')
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


if __name__ == '__main__':
    np.set_printoptions(threshold=5000)

    static_1 = SystemStaticParameters(
        init_type=1,
        lattice_constant=1.75,
        particles_number=(7, 7, 7),
        # particles_number=(2, 2, 2),
        # particles_number=1372,
        crystal_type='гцк',
    )
    external_1 = ExternalParameters(
        environment_type='velocity_scaling',
        heating_velocity=0.02,
        temperature=2.8,
    )
    # INITIAL_TIME_STEP = 0.005
    # INITIAL_ITERATION_NUMBERS = 40000
    INITIAL_TIME_STEP = 0.001
    INITIAL_ITERATION_NUMBERS = 200000
    model_1 = ModelingParameters(
        iterations_numbers=INITIAL_ITERATION_NUMBERS,
        # iterations_numbers=1000,
        time_step=INITIAL_TIME_STEP,
        initial_temperature=1e-5,
    )
    dynamic_1 = SystemDynamicParameters(
        static=static_1,
        # temperature=model_1.initial_temperature,
    )
    potential_1 = PotentialParameters(
        potential_type='lennard_jones',
        skin=0.3,
    )
    md_sample_1 = MolecularDynamics(
        static=static_1,
        dynamic=dynamic_1,
        external=external_1,
        model=model_1,
        potential=potential_1,
    )
    # run(
    #     'md_sample_1.run_md()',
    #     sort=2,
    # )
    md_sample_1.run_md()
    # run('md_sample_1.run_rdf(50)', sort=2)
