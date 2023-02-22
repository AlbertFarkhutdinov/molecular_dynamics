from pretty_repr import RepresentableObject


class SimulationParameters(RepresentableObject):

    def __init__(self, **simulation_kwargs):
        self.iterations_numbers = simulation_kwargs[
            'iterations_numbers'
        ]
        self.configuration_storing_step = simulation_kwargs[
            'configuration_storing_step'
        ]
        self.configuration_saving_step = simulation_kwargs[
            'configuration_saving_step'
        ]
        self.ensembles_number = simulation_kwargs[
            'ensembles_number'
        ]
        self.isotherm_saving_step = simulation_kwargs.get(
            'isotherm_saving_step',
        )
        self.equilibration_steps = simulation_kwargs.get('equilibration_steps')
        self.ssf_steps = simulation_kwargs.get('ssf_steps')
        self.ssf_max_wave_number = simulation_kwargs.get('ssf_max_wave_number')

    def __str__(self):
        return repr(self)
