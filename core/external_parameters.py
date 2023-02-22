import numpy as np
from pretty_repr import RepresentableObject


class ExternalParameters(RepresentableObject):

    def __init__(self, **external_kwargs):
        self.heating_velocity = external_kwargs.get('heating_velocity')
        self.environment_type = external_kwargs.get('environment_type')
        self.temperature = external_kwargs.get('temperature')
        self.pressure = external_kwargs.get('pressure')
        self.thermostat_parameters = np.array(
            external_kwargs.get('thermostat_parameters')
        )
        self.barostat_parameter = external_kwargs.get('barostat_parameter')

    def __str__(self):
        string = ''
        if self.temperature is not None:
            string += f'T_{self.temperature:.5f}_'
        if self.pressure is not None:
            string += f'P_{self.pressure:.5f}_'
        if self.heating_velocity is not None:
            string += f'HV_{self.heating_velocity:.5f}_'
        return string
