class ModelingParameters:

    def __init__(
            self,
            iterations_numbers: int,
            time_step: float,
            initial_temperature: float,
    ):
        self.time = 0
        self.iterations_numbers = iterations_numbers
        self.time_step = time_step
        self.initial_temperature = initial_temperature
