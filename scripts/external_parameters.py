class ExternalParameters:

    def __init__(
            self,
            temperature: float,
            heating_velocity: float,
            environment_type: str,
            pressure: float = None,
            **kwargs,
    ):
        self.temperature = temperature
        self.pressure = pressure
        self.heating_velocity = heating_velocity
        self.environment_type = environment_type
        if self.environment_type == 'nose_hoover':
            self.thermostat_parameter = kwargs.get('thermostat_parameter')
            self.barostat_parameter = kwargs.get('barostat_parameter')
