from cProfile import run
from pstats import Stats
from datetime import datetime
from math import pi

import matplotlib.pyplot as plt
import numpy as np

from dynamic_parameters import SystemDynamicParameters
from external_parameters import ExternalParameters
from helpers import get_empty_float_scalars, get_empty_int_scalars, get_empty_vectors
from modeling_parameters import ModelingParameters
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

    def system_dynamics(
            self,
            stage_id: int,
            thermostat_type: str,
            **kwargs,
    ):
        if thermostat_type == 'velocity_scaling':
            if stage_id == 1:
                self.velocity_scaling_1()
            if stage_id == 2:
                self.velocity_scaling_2(**kwargs)
        elif thermostat_type == 'nose_hoover':
            if stage_id == 1:
                self.nose_hoover_1()
            if stage_id == 2:
                self.nose_hoover_2(**kwargs)

    def velocity_scaling_1(self):
        self.dynamic.get_next_positions(self.model.time_step)
        temperature = self.dynamic.temperature
        # print(f'{temperature=}')
        # if temperature == 0:
        #     temperature = self.model.initial_temperature
        # temperature += (
        #         (1 if (self.external.temperature - temperature) >= 0 else -1)
        #         * self.external.heating_velocity
        #         * self.model.time_step
        # )
        # self.dynamic.get_next_velocities(
        #     time_step=self.model.time_step,
        #     # vel_coefficient=np.sqrt(self.external.temperature / temperature),
        #     vel_coefficient=np.sqrt(self.model.initial_temperature / temperature),
        # )
        self.model.initial_temperature += (
                (1 if (self.external.temperature - self.model.initial_temperature) >= 0 else -1)
                * self.external.heating_velocity
                * self.model.time_step
        )
        if temperature == 0:
            temperature = self.model.initial_temperature
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
            vel_coefficient=np.sqrt(self.model.initial_temperature / temperature),
        )

    def velocity_scaling_2(
            self,
            temperature: float = None,
            virial: float = None,
            system_kinetic_energy: float = None,
            potential_energy: float = None,
            pressure: float = None,
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
        self.dynamic.get_next_velocities(
            time_step=self.model.time_step,
        )
        total_energy = _system_kinetic_energy + _potential_energy
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
                ) * self.model.time_step ** 2 / nose_hoover_thermostat_parameter
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
        error = 1e-5
        p_s = self.external.parameters['lambda']
        ready = False
        _iter = 0
        while not ready and _iter < 50:
            _iter += 1
            p_o = p_s
            ds = 0
            pass  # TODO

    def load_forces(
            self,
            potential_table: np.ndarray,
    ):
        potential_energies = get_empty_float_scalars(self.static.particles_number)
        virial = 0
        distances = self.dynamic.interparticle_distances
        if self.potential.update_test:
            self.update_list(distances=distances)
            self.potential.update_test = False
        for i in range(self.static.particles_number - 1):
            for k in range(
                    self.verlet_lists['marker_1'][i],
                    self.verlet_lists['marker_2'][i] + 1,
            ):
                j = self.verlet_lists['list'][k]
                distance = distances[i][j]
                if distance < self.potential.r_cut:
                    table_row = int((distance - 0.5) / 0.0001)
                    potential_ij = potential_table[table_row, 0]
                    force_ij = potential_table[table_row, 1]
                    potential_energies[j] += potential_ij
                    virial += force_ij * distance
                    acceleration_ij = (
                            force_ij
                            * (self.dynamic.positions[i] - self.dynamic.positions[j])
                            / distance
                    )
                    self.dynamic.accelerations[i] += acceleration_ij
                    self.dynamic.accelerations[j] -= acceleration_ij
        potential_energy = potential_energies.sum()
        self.load_move_test()
        return potential_energy, virial

    def update_list(self, distances: np.ndarray):
        advances = (distances < (self.potential.r_cut + self.potential.skin)).astype(np.int)
        k = 1
        for i in range(self.static.particles_number - 1):
            self.verlet_lists['marker_1'][i] = k
            for j in range(i + 1, advances[i].nonzero()[0][-1] + 1):
                self.verlet_lists['list'][k] = j
                if advances[i][j]:
                    k += advances[i][j]
            self.verlet_lists['marker_2'][i] = k - 1

    def load_move_test(self):
        displacements = (
                self.dynamic.velocities * self.model.time_step
                + self.dynamic.accelerations * self.model.time_step ** 2 / 2.0
        )
        ds_1, ds_2 = 0, 0
        for i in range(self.static.particles_number):
            ds = (displacements[i] ** 2).sum() ** 0.5
            if ds >= ds_1:
                ds_2 = ds_1
                ds_1 = ds
            elif ds >= ds_2:
                ds_2 = ds
        self.potential.update_test = (ds_1 + ds_2) > self.potential.skin

    def save_config(self, file_name: str = None):
        file_name = file_name or 'system_config.txt'
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
        file_name = file_name or 'system_config.txt'
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
        parameters = {
            'temperature': _temperature,
            'pressure': _pressure,
            'kinetic_energy': _system_kinetic_energy,
            'potential_energy': _potential_energy,
            'total_energy': _system_kinetic_energy + _potential_energy,
        }
        for key, value in parameters.items():
            with open(f'system_{key}.txt', mode='a', encoding='utf-8') as file:
                file.write(f'{self.model.time} {value}\n')

    def load_lammps_trajectory(self, file_name: str = None):
        file_name = file_name or 'system_config.txt'
        with open(file_name, mode='a', encoding='utf-8') as file:
            lines = [
                'ITEM: TIMESTEP',
                str(self.model.time_step),
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
            file.writelines('\n'.join(lines))

    def run_md(self):
        # self.load_save_config()
        for step in range(1, self.model.iterations_numbers + 1):
            self.model.time += self.model.time_step
            self.system_dynamics(
                stage_id=1,
                thermostat_type='velocity_scaling',
            )
            self.dynamic.boundary_conditions()
            system_kinetic_energy = self.dynamic.system_kinetic_energy
            temperature = self.dynamic.temperature
            potential_energy, virial = self.load_forces(
                potential_table=self.potential.potential_table,
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
            # self.save_config(file_name=f'system_config_{step}.txt')
            self.save_system_parameters(
                **parameters
            )
            self.load_lammps_trajectory()
            print(
                f'Step: {step}/{self.model.iterations_numbers};',
                f'\tTime = {self.model.time};',
                f'\tT = {temperature:.30f};',
                f'\tP = {pressure:.30f};',
                'Calculation completed.\n',
                sep='\n',
            )

    def run_rdf(self, steps_number: int = 1000):
        layer_thickness = 0.01
        begin_step = 1
        end_step = steps_number
        rdf = get_empty_float_scalars(10 * self.static.particles_number)
        with open('system_config.txt', mode='r', encoding='utf8') as file:
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
                        / (4.0 * pi * radiuses ** 2 * self.static.particles_number ** 2)
                        * particles_in_layer / layer_thickness
                )
                print(f'Step: {step}/{end_step};')
        with open('rdf_file.txt', mode='w', encoding='utf8') as file:
            rdf = rdf[:np.nonzero(rdf)[0][-1]]
            radiuses = layer_thickness * (np.arange(rdf.size) + 0.5)
            rdf = rdf / (end_step - begin_step + 1)
            # mask = (radiuses <= self.static.cell_dimensions[0] / 2.0)
            for i, radius in enumerate(radiuses):
                if radius <= self.static.cell_dimensions[0] / 2.0:
                    file.write(f'{radius} {rdf[i]}\n')

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

    start = datetime.now()
    # Liquid argon:
    # mass = 6.69E-26 kilograms
    # distance = sigma = 0.341E-9 meters
    # energy (temperature) = epsilon = 1.65E-21 joules (119.8 kelvin)
    # tau = sigma * sqrt(mass / epsilon) = 2.17E-12 seconds
    # velocity = sigma / tau = 1.57E2 m/s
    # force = epsilon/sigma = 4.85E-12 newtons
    # pressure = epsilon / (sigma ^ 3) = 4.2E7 pascal = 4.2E2 atmospheres

    static_1 = SystemStaticParameters(
        init_type=1,
        lattice_constant=1.75,
        particles_number=(7, 7, 7),
        # particles_number=1372,
        crystal_type='гцк',
    )
    dynamic_1 = SystemDynamicParameters(
        static=static_1,
    )
    external_1 = ExternalParameters(
        environment_type='velocity_scaling',
        heating_velocity=0.02,
        temperature=2.8,
    )
    model_1 = ModelingParameters(
        # iterations_numbers=40000,
        iterations_numbers=1,
        time_step=0.005,
        initial_temperature=2.8,
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
    # run('md_sample_1.run_md()', sort=2)
    md_sample_1.run_md()
    run('md_sample_1.run_rdf(50)', sort=2)
    print(datetime.now() - start)
