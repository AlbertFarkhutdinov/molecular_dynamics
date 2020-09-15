class ExternalParameters:

    def __init__(
            self,
            temperature: float,
            heating_velocity: float,
            environment_type: str,
            pressure: float = None,
            **environment_parameters,
    ):
        self.temperature = temperature
        self.pressure = pressure
        self.heating_velocity = heating_velocity
        if environment_type == 'nose_hoover':
            self.parameters = environment_parameters
