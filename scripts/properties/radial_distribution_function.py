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
        self.radii = get_empty_float_scalars(
            20 * sample.system.configuration.particles_number
        )

    def calculate(self):
        static_distances = self.sample.interparticle_distances.flatten()
        self.radii = np.arange(
            self.layer_thickness,
            static_distances.max() + 1,
            self.layer_thickness,
        )
        rdf_hist = np.histogram(static_distances.flatten(), self.radii)[0]
        return (
                2.0 * self.sample.system.volume
                / (
                        4.0 * np.pi * self.radii[:rdf_hist.size] ** 2
                        * self.sample.system.configuration.particles_number
                        * self.sample.system.configuration.particles_number
                ) * rdf_hist / self.layer_thickness
        )

    def accumulate(self):
        _rdf_sample = self.calculate()

        # TODO
        # self.save_sample(_rdf_sample)

        self.rdf[:_rdf_sample.size] += _rdf_sample

    def normalize(self):
        self.rdf = (
                self.rdf[:np.nonzero(self.rdf)[0][-1]] / self.ensembles_number
        )
        self.radii = self.layer_thickness * np.arange(1, self.rdf.size + 1)

    def save_sample(self, sample):
        _radii = self.layer_thickness * np.arange(1, sample.size + 1)
        _is_saved_radii = _radii <= self.sample.system.cell_dimensions[0] / 2.0
        Saver(simulation_parameters=self.sample.sim_parameters).save_rdf(
            rdf_data={
                'radius': _radii[_is_saved_radii],
                'rdf': sample[_is_saved_radii],
            },
            file_name=(
                f'rdf_sample_{str(self.sample.externals)}_'
                f'{self.sample.get_str_time()}.csv'
            )
        )

    def save(self):
        self.normalize()
        _is_saved = self.radii <= self.sample.system.cell_dimensions[0] / 2.0
        Saver(simulation_parameters=self.sample.sim_parameters).save_rdf(
            rdf_data={
                'radius': self.radii[_is_saved],
                'rdf': self.rdf[_is_saved],
            },
            file_name=(
                f'rdf_{str(self.sample.externals)}.csv'
            )
        )
