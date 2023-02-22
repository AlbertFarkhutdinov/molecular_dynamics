from time import time

from scripts.log_config import log_debug_info
from scripts.properties.radial_distribution_function import RadialDistributionFunction
from scripts.properties.transport_properties import TransportProperties


class Isotherm:

    def __init__(
            self,
            sample,
            layer_thickness: float = 0.01,
    ):
        """
        Ensembles number for static structure properties (RDF)
        is 2 * `ensembles_number` - 1.
        Ensembles number for dynamic properties (MSD, etc.)
        is `ensembles_number`.

        """
        self.start = time()
        self.sample = sample
        self.layer_thickness = layer_thickness
        sample.fix_external_conditions()
        sample.equilibrate_system(
            equilibration_steps=sample.sim_parameters.equilibration_steps,
        )
        self.ensembles_number = sample.sim_parameters.ensembles_number
        self.steps_number = 2 * self.ensembles_number - 1
        self.rdf = RadialDistributionFunction(
            sample=self.sample,
            layer_thickness=layer_thickness,
            ensembles_number=self.ensembles_number
        )
        self.transport_properties = TransportProperties(
            sample=self.sample,
        )
        # self.ssf = StaticStructureFactor(
        #     sample=self.sample,
        #     max_wave_number=sample.sim_parameters.ssf_max_wave_number,
        #     ensembles_number=self.ensembles_number,
        #     layer_thickness=0.01 * layer_thickness,
        # )

    def print_current_state(self, step):
        temperature = self.sample.system.configuration.temperature
        volume = self.sample.system.volume
        density = self.sample.system.get_density(volume=volume)
        pressure = self.sample.system.get_pressure(
            temperature=temperature,
            volume=volume,
            density=density,
        )
        message = (
            f'Isotherm Step: {step}/{self.steps_number}, '
            f'Temperature = {temperature:8.5f} epsilon/k_B, \t'
            f'Pressure = {pressure:.5f} epsilon/sigma^3, \t'
        )
        log_debug_info(message)
        print(message)

    def run(self):
        print('********Isothermal calculations started********')
        for step in range(1, self.steps_number + 1):
            self.print_current_state(step)
            self.sample.md_time_step(
                step=step,
                is_rdf_calculation=True,
                is_pbc_switched_on=False,
            )
            self.transport_properties.step = step
            if step <= self.ensembles_number:
                self.transport_properties.init_ensembles()
                self.sample.calculate_interparticle_vectors()
                # if step <= self.sample.sim_parameters.ssf_steps:
                #     self.ssf.accumulate()
                self.rdf.accumulate()

                # TODO
                # self.sample.saver.save_configuration(
                #     file_name=(
                #         'system_configuration_'
                #         f'{self.sample.get_str_time()}.csv'
                #     ),
                # )

            self.transport_properties.accumulate()

        self.transport_properties.save()
        self.rdf.save()
        # self.ssf.save()
        print(
            f'Equilibrium dynamics simulation is completed. '
            f'Time of calculation: {time() - self.start} seconds.'
        )
