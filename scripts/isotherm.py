from time import time

from scripts.log_config import log_debug_info
from scripts.radial_density_function import RadialDensityFunction
from scripts.transport_properties import TransportProperties
from scripts.static_structure_factor import StaticStructureFactor


class Isotherm:

    def __init__(
            self,
            sample,
            layer_thickness: float = 0.01,
    ):
        self.start = time()
        self.sample = sample
        self.layer_thickness = layer_thickness
        sample.fix_external_conditions()
        sample.equilibrate_system(
            equilibration_steps=sample.isotherm_parameters['equilibration_steps'],
        )
        self.ensembles_number = sample.isotherm_parameters['ensembles_number']
        self.steps_number = 2 * self.ensembles_number - 1
        self.rdf = RadialDensityFunction(
            sample=self.sample,
            layer_thickness=layer_thickness,
            ensembles_number=self.ensembles_number
        )
        self.transport_properties = TransportProperties(
            sample=self.sample,
        )
        self.ssf = StaticStructureFactor(
            sample=self.sample,
            max_wave_number=4,
            ensembles_number=self.ensembles_number,
            layer_thickness=0.01 * layer_thickness,
        )

    def print_current_state(self, step):
        temperature = self.sample.dynamic.temperature()
        pressure = self.sample.dynamic.get_pressure(
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
            self.sample.md_time_step(
                potential_table=self.sample.potential.potential_table,
                step=step,
                is_rdf_calculation=True,
            )
            self.transport_properties.step = step
            if step <= self.ensembles_number:
                self.transport_properties.init_ensembles()
                self.sample.dynamic.calculate_interparticle_vectors()
                self.rdf.accumulate()
                self.ssf.accumulate()

            self.transport_properties.acccumulate()
        self.transport_properties.save()
        self.rdf.save()
        self.ssf.save()
        print(f'Calculation completed. Time of calculation: {time() - self.start} seconds.')
