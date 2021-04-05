import numpy as np

from scripts_new.helpers import get_representation


class ExternalParameters:

    def __init__(self, **external_kwargs):
        self.heating_velocity = external_kwargs['heating_velocity']
        self.environment_type = external_kwargs.get('environment_type')
        self.temperature = external_kwargs.get('temperature')
        self.pressure = external_kwargs.get('pressure')
        if self.environment_type in ('nose_hoover', 'mttk'):
            self.thermostat_parameters = np.array(
                external_kwargs.get('thermostat_parameters')
            )
            self.barostat_parameter = external_kwargs.get('barostat_parameter')

    def __repr__(self):
        return get_representation(self)

    def __str__(self):
        return (
            f'T_{self.temperature:.5f}_'
            f'P_{self.pressure:.5f}_'
            f'{self.heating_velocity:.8f}'
        )
