from time import time

from scripts.log_config import log_debug_info
from scripts.radial_distribution_function import RadialDistributionFunction
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
        #     max_wave_number=sample.isotherm_parameters['ssf_max_wave_number'],
        #     ensembles_number=self.ensembles_number,
        #     layer_thickness=0.01 * layer_thickness,
        # )

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
                is_pbc_switched_on=False,
            )
            self.transport_properties.step = step
            if step <= self.ensembles_number:
                self.transport_properties.init_ensembles()
                self.sample.dynamic.calculate_interparticle_vectors()
                self.rdf.accumulate()
                # if step <= self.sample.isotherm_parameters['ssf_steps']:
                #     self.ssf.accumulate()

            self.transport_properties.accumulate()
            _file_name = f'T_{self.sample.verlet.external.temperature:.5f}_dt_{self.sample.model.time_step}.xyz'
            self.sample.dynamic.save_xyz_file(
                filename=_file_name,
                step=step,
            )

        self.transport_properties.save()
        self.rdf.save()
        # self.ssf.save()
        print(f'Calculation completed. Time of calculation: {time() - self.start} seconds.')
