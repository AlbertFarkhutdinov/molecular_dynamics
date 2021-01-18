from copy import deepcopy
from time import time

# import numpy as np

from scripts.helpers import get_parameters_dict
from scripts.log_config import log_debug_info
from scripts.radial_density_function import RadialDensityFunction
from scripts.saver import Saver
# from scripts.static_structure_factor import StaticStructureFactor


class Isotherm:

    def __init__(
            self,
            sample,
            virial,
            layer_thickness: float = 0.01,
    ):
        self.start = time()
        self.sample = sample
        self.virial = virial
        self.layer_thickness = layer_thickness
        sample.fix_external_conditions(
            virial=virial,
        )
        sample.equilibrate_system(
            equilibration_steps=sample.isotherm_parameters['equilibration_steps'],
        )
        self.ensembles_number = sample.isotherm_parameters['ensembles_number']
        self.steps_number = 2 * self.ensembles_number - 1
        self.isotherm_system_parameters = get_parameters_dict(
            names=(
                'time',
                'msd',
                'einstein_diffusion',
                'velocity_autocorrelation',
                'green_kubo_diffusion',
            ),
            value_size=self.steps_number,
        )
        self.first_positions, self.first_velocities = {}, {}
        self.green_kubo_diffusion = 0
        self.rdf = RadialDensityFunction(
            sample=self.sample,
            layer_thickness=layer_thickness,
            ensembles_number=self.ensembles_number
        )
        # self.ssf = StaticStructureFactor(
        #     sample=self.sample,
        #     max_wave_number=5,
        #     layer_thickness=0.01 * layer_thickness,
        # )

        # van_hove = np.array([
        #     get_empty_float_scalars(20 * sample.static.particles_number) for _ in range(ensembles_number)
        # ])

    def print_current_state(self, step):
        temperature = self.sample.dynamic.temperature()
        pressure = self.sample.dynamic.get_pressure(
            virial=self.virial,
            temperature=temperature,
            cell_volume=self.sample.static.get_cell_volume(),
            density=self.sample.static.get_density()
        )
        message = (
            f'Isotherm Step: {step}/{self.steps_number}, '
            f'Temperature = {temperature:8.5f} epsilon/k_B, \t'
            f'Pressure = {pressure:.5f} epsilon/sigma^3, \t'
        )
        log_debug_info(message)
        print(message)

    def run(self):
        print(f'********Isothermal calculations started********')
        for step in range(1, self.steps_number + 1):
            self.print_current_state(step)
            self.virial = self.sample.md_time_step(
                potential_table=self.sample.potential.potential_table,
                step=step,
                is_rdf_calculation=True,
            )

            first_step = step - self.ensembles_number
            if step <= self.ensembles_number:
                first_step = 0
                self.first_positions[step] = deepcopy(self.sample.dynamic.positions)
                self.first_velocities[step] = deepcopy(self.sample.dynamic.velocities)
                self.isotherm_system_parameters['time'][step - 1] = self.sample.model.time_step * step
                self.sample.dynamic.calculate_interparticle_vectors()

                self.rdf.accumulate()
                # self.ssf.accumulate()

                # TODO implement Van-Hove function, scattering function, dynamic structure factor,

            for i in range(first_step, step):
                self.isotherm_system_parameters['msd'][i] += self.sample.dynamic.get_msd(
                    previous_positions=self.first_positions[step - i],
                )
                self.isotherm_system_parameters['velocity_autocorrelation'][i] += (
                        (self.first_velocities[step - i] * self.sample.dynamic.velocities).sum()
                        / self.sample.static.particles_number
                )

        for key, value in self.isotherm_system_parameters.items():
            self.isotherm_system_parameters[key] = value[:self.ensembles_number]

        self.isotherm_system_parameters['msd'] = self.isotherm_system_parameters['msd'] / self.ensembles_number
        self.isotherm_system_parameters[
            'velocity_autocorrelation'
        ] = self.isotherm_system_parameters['velocity_autocorrelation'] / self.ensembles_number

        self.isotherm_system_parameters[
            'einstein_diffusion'
        ] = self.isotherm_system_parameters['msd'] / 6 / self.isotherm_system_parameters['time']

        for i in range(self.ensembles_number):
            self.green_kubo_diffusion += self.isotherm_system_parameters[
                                        'velocity_autocorrelation'
                                    ][i] * self.sample.model.time_step / 3
            self.isotherm_system_parameters['green_kubo_diffusion'][i] += self.green_kubo_diffusion

        self.rdf.save()
        # self.ssf.save()
        Saver().save_dict(
            data=self.isotherm_system_parameters,
            default_file_name=f'transport.csv',
            data_name='MSD and self-diffusion coefficient',
            file_name=f'transport_T_{self.sample.verlet.external.temperature:.5f}.csv'
        )
        print(f'Calculation completed. Time of calculation: {time() - self.start} seconds.')
