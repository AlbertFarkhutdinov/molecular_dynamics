from atooms.trajectory import Trajectory
import atooms.postprocessing as pp
import numpy as np

from scripts.helpers import get_empty_float_scalars
from scripts.numba_procedures import get_static_structure_factors
from scripts.numba_procedures import get_unique_ssf
from scripts.saver import Saver


class StaticStructureFactor:

    def __init__(
            self,
            sample,
            max_wave_number,
            ensembles_number,
            layer_thickness: float = 0.0001,
    ):
        self.sample = sample
        self.layer_thickness = layer_thickness
        self.ensembles_number = ensembles_number
        self.wave_numbers_range = np.arange(
            0,
            max_wave_number + 1,
            self.layer_thickness,
        )
        _components_range = np.arange(max_wave_number + 1, dtype=np.float32)
        self.all_wave_vectors = np.array([
            [i, j, k]
            for i in _components_range
            for j in _components_range
            for k in _components_range
        ]) * 2 * np.pi / sample.system.cell_dimensions
        self.all_wave_numbers = (self.all_wave_vectors ** 2).sum(axis=1) ** 0.5
        self.ssf = get_empty_float_scalars(self.wave_numbers_range.size)

    @staticmethod
    def calculate_by_atooms(trajectory_file_path):
        ssf_instance = pp.StructureFactor(
            Trajectory(trajectory_file_path),
            # kmin=0.0,
            # kmax=20.0,
            # ksamples=1000,
            # nk=20,
            # dk=0.1,
        )
        ssf_instance.do()
        return ssf_instance.grid, ssf_instance.value

    def accumulate(self):
        srv = self.sample.system.configuration.interparticle_vectors
        srv = srv.reshape(
            int(srv.size / 3),
            3,
        )
        static_structure_factors = get_static_structure_factors(
            wave_vectors=self.all_wave_vectors,
            static_radius_vectors=srv,
            particles_number=self.sample.configuration.particles_number,
        )
        wave_numbers, static_structure_factors = get_unique_ssf(
            wave_numbers=self.all_wave_numbers,
            static_structure_factors=static_structure_factors,
            layer_thickness=self.layer_thickness,
        )
        k_hist = np.histogram(
            wave_numbers,
            self.wave_numbers_range
        )[0]
        self.ssf[:k_hist.size][k_hist != 0] += static_structure_factors

    def normalize(self):
        self.ssf = self.ssf / self.ensembles_number

    def save(self):
        self.normalize()
        Saver(
            simulation_parameters=self.sample.sim_parameters,
        ).save_dict(
            data={
                'wave_number': self.wave_numbers_range,
                'static_structure_factor': self.ssf,
            },
            default_file_name='static_structure_factor.csv',
            data_name='Static Structure Factor',
            file_name=(
                f'static_structure_factor_{str(self.sample.externals)}.csv'
            )
        )
