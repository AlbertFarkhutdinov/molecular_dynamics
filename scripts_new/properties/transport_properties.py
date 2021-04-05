from copy import deepcopy

from scripts_new.helpers import get_parameters_dict
from scripts_new.saver import Saver


class TransportProperties:

    def __init__(
            self,
            sample,
    ):
        self.sample = sample
        self.ensembles_number = sample.sim_parameters.ensembles_number
        self.step = 1
        self.first_positions, self.first_velocities = {}, {}
        self.green_kubo_diffusion = 0
        self.data = get_parameters_dict(
            names=(
                'time',
                'msd',
                'einstein_diffusion',
                'velocity_autocorrelation',
                'green_kubo_diffusion',
            ),
            value_size=2 * self.ensembles_number - 1,
        )
        # TODO implement Van-Hove function, scattering function, dynamic structure factor

    def init_ensembles(self):
        self.first_positions[self.step] = deepcopy(self.sample.system.configuration.positions)
        self.first_velocities[self.step] = deepcopy(self.sample.system.configuration.velocities)
        self.data['time'][self.step - 1] = self.sample.immutables.time_step * self.step

    def accumulate(self):
        first_step = 0 if self.step <= self.ensembles_number else self.step - self.ensembles_number
        for i in range(first_step, self.step):
            self.data['msd'][i] += self.sample.system.configuration.get_msd(
                previous_positions=self.first_positions[self.step - i],
            )
            self.data['velocity_autocorrelation'][i] += (
                    (self.first_velocities[self.step - i] * self.sample.system.configuration.velocities).sum()
                    / self.sample.system.configuration.particles_number
            )

    def normalize(self):
        for key, value in self.data.items():
            self.data[key] = value[:self.ensembles_number]

        self.data['msd'] = self.data['msd'] / self.ensembles_number
        self.data[
            'velocity_autocorrelation'
        ] = self.data['velocity_autocorrelation'] / self.ensembles_number
        self.data[
            'einstein_diffusion'
        ] = self.data['msd'] / 6.0 / self.data['time']

        for i in range(self.ensembles_number):
            self.green_kubo_diffusion += self.data[
                                        'velocity_autocorrelation'
                                    ][i] * self.sample.immutables.time_step / 3
            self.data['green_kubo_diffusion'][i] += self.green_kubo_diffusion

    def save(self):
        self.normalize()
        Saver(
            simulation_parameters=self.sample.sim_parameters,
        ).save_dict(
            data=self.data,
            default_file_name='transport.csv',
            data_name='MSD and self-diffusion coefficient',
            file_name=(
                f'transport{str(self.sample.externals)}.csv'
            )
        )
