import numpy as np

from scripts.potentials.base_potential import BasePotential

from scripts.helpers import get_empty_float_scalars


class Dzugutov(BasePotential):

    def __init__(self):
        super().__init__()
        self.r_cut = 1.94
        self.table_size = 15000
        self.distances = 0.5 + 0.0001 * np.arange(1, self.table_size + 1)
        self.potential_table = self.get_energies_and_forces()

    def get_energies_and_forces(self):
        aa, bb, a, c, d, m = 5.82, 1.28, 1.87, 1.1, 0.27, 16
        v_1 = get_empty_float_scalars(self.table_size)
        v_2 = get_empty_float_scalars(self.table_size)
        f_1 = get_empty_float_scalars(self.table_size)
        f_2 = get_empty_float_scalars(self.table_size)

        for i in range(self.table_size):
            if self.distances[i] / self.sigma < a:
                _aa_eps = aa * self.epsilon
                _sigma_dist = self.sigma / self.distances
                _exp_arg = self.sigma * c / (self.distances - self.sigma * a)
                v_1[i] = _aa_eps * (_sigma_dist ** m - bb) * np.exp(_exp_arg)
                f_1[i] = (
                        _aa_eps
                        * (
                                -m / self.sigma * _sigma_dist ** (m + 1)
                                - _sigma_dist ** m - bb
                        ) * np.exp(_exp_arg) / _exp_arg / _exp_arg
                )
            if self.distances[i] / self.sigma < self.r_cut:
                _bb_eps = bb * self.epsilon
                _sigma_cut = self.sigma * self.r_cut
                _exp_arg = _sigma_cut / (self.distances - _sigma_cut)
                v_2[i] = _bb_eps * np.exp(_exp_arg)
                f_2[i] = (
                        -_bb_eps * self.sigma * d * np.exp(_exp_arg)
                        / (self.distances - _sigma_cut)
                        / (self.distances - _sigma_cut)
                )

        potential_table = np.array(
            [v_1 + v_2, -1.0 * (f_1 + f_2)],
            dtype=np.float
        ).transpose()
        return potential_table
