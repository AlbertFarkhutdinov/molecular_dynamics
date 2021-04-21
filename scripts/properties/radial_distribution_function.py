from atooms.trajectory import Trajectory
import atooms.postprocessing as pp
import numpy as np

from scripts.helpers import get_empty_float_scalars
from scripts.saver import Saver


class RadialDistributionFunction:

    def __init__(
            self,
            sample,
            ensembles_number,
            layer_thickness: float = 0.01,
    ):
        self.sample = sample
        self.layer_thickness = layer_thickness
        self.ensembles_number = ensembles_number
        self.rdf = get_empty_float_scalars(
            20 * sample.system.configuration.particles_number
        )
        self.radiuses = get_empty_float_scalars(
            20 * sample.system.configuration.particles_number
        )

    def calculate_by_atooms(self, trajectory_file_path):
        rdf_instance = pp.RadialDistributionFunction(
            Trajectory(trajectory_file_path),
            norigins=self.ensembles_number,
            dr=self.layer_thickness,
        )
        rdf_instance.do()
        return rdf_instance.grid, rdf_instance.value

    def accumulate(self):
        static_distances = self.sample.interparticle_distances.flatten()
        self.radiuses = np.arange(
            self.layer_thickness,
            static_distances.max() + 1,
            self.layer_thickness,
        )
        rdf_hist = np.histogram(
            static_distances.flatten(),
            self.radiuses
        )[0]
        self.rdf[:rdf_hist.size] += (
                2.0 * self.sample.system.volume
                / (4.0 * np.pi * self.radiuses[:rdf_hist.size] ** 2
                   * self.sample.system.configuration.particles_number
                   * self.sample.system.configuration.particles_number)
                * rdf_hist / self.layer_thickness
        )

    def normalize(self):
        self.rdf = self.rdf[
                   :np.nonzero(self.rdf)[0][-1]
                   ] / self.ensembles_number
        self.radiuses = self.layer_thickness * np.arange(1, self.rdf.size + 1)

    def save(self):
        self.normalize()
        Saver(
            simulation_parameters=self.sample.sim_parameters,
        ).save_rdf(
            rdf_data={
                'radius': (
                    self.radiuses[
                        self.radiuses
                        <= self.sample.system.cell_dimensions[0] / 2.0
                    ]
                ),
                'rdf': self.rdf[
                    self.radiuses
                    <= self.sample.system.cell_dimensions[0] / 2.0
                ],
            },
            file_name=(
                f'rdf_{str(self.sample.externals)}.csv'
            )
        )
