import numpy as np


class ExternalParameters:

    def __init__(
            self,
            heating_velocity: float,
            environment_type: str,
            temperature: float,
            pressure: float = None,
            **kwargs,
    ):
        self.temperature = temperature
        self.pressure = pressure
        self.heating_velocity = heating_velocity
        self.environment_type = environment_type
        if self.environment_type in ('nose_hoover', 'mtk'):
            self.thermostat_parameters = np.array(kwargs.get('thermostat_parameters'))
            self.barostat_parameter = kwargs.get('barostat_parameter')
