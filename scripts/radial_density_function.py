import numpy as np

from scripts.helpers import get_empty_float_scalars
from scripts.saver import Saver


class RadialDensityFunction:

    def __init__(
            self,
            sample,
            ensembles_number,
            layer_thickness: float = 0.01,
    ):
        self.sample = sample
        self.layer_thickness = layer_thickness
        self.ensembles_number = ensembles_number
        self.rdf = get_empty_float_scalars(20 * sample.static.particles_number)

    def accumulate(self):
        static_distances = self.sample.dynamic.interparticle_distances.flatten()
        radiuses = np.arange(self.layer_thickness, static_distances.max() + 1, self.layer_thickness)
        rdf_hist = np.histogram(
            static_distances.flatten(),
            radiuses
        )[0]
        self.rdf[:rdf_hist.size] += (
                2.0 * self.sample.static.get_cell_volume()
                / (4.0 * np.pi * radiuses[:rdf_hist.size] ** 2
                   * self.sample.static.particles_number * self.sample.static.particles_number)
                * rdf_hist / self.layer_thickness
        )

    def save(self):
        self.rdf = self.rdf[:np.nonzero(self.rdf)[0][-1]] / self.ensembles_number
        radiuses = self.layer_thickness * np.arange(1, self.rdf.size + 1)
        Saver().save_rdf(
            rdf_data={
                'radius': radiuses[radiuses <= self.sample.static.cell_dimensions[0] / 2.0],
                'rdf': self.rdf[radiuses <= self.sample.static.cell_dimensions[0] / 2.0],
            },
            file_name=f'rdf_T_{self.sample.verlet.external.temperature:.5f}.csv'
        )