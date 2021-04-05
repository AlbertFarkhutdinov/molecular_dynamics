import numpy as np

from scripts_new.potentials.base_potential import BasePotential

from scripts_new.helpers import get_empty_float_scalars


class Dzugutov(BasePotential):

    def __repr__(self):
        return f'{self.__class__.__name__}(table_size = {self.table_size!r})'

    def get_energies_and_forces(self):
        factors = (5.82, 1.28, 1.87, 1.1, 0.27, 16)
        v_1 = get_empty_float_scalars(self.table_size)
        v_2 = get_empty_float_scalars(self.table_size)
        f_1 = get_empty_float_scalars(self.table_size)
        f_2 = get_empty_float_scalars(self.table_size)

        for i in range(self.table_size):
            if self.distances[i] / self.sigma < factors[2]:
                _a_eps = factors[0] * self.epsilon
                _sigma_dist = self.sigma / self.distances
                _exp_arg = (
                        self.sigma * factors[3]
                        / (self.distances - self.sigma * factors[2])
                )
                v_1[i] = (
                        _a_eps
                        * (_sigma_dist ** factors[5] - factors[1])
                        * np.exp(_exp_arg)
                )
                f_1[i] = (
                        _a_eps
                        * (
                                -factors[5] / self.sigma
                                * _sigma_dist ** (factors[5] + 1)
                                - _sigma_dist ** factors[5] - factors[1]
                        ) * np.exp(_exp_arg) / _exp_arg / _exp_arg
                )
            if self.distances[i] / self.sigma < self.r_cut:
                _b_eps = factors[1] * self.epsilon
                _sigma_cut = self.sigma * self.r_cut
                _exp_arg = _sigma_cut / (self.distances - _sigma_cut)
                v_2[i] = _b_eps * np.exp(_exp_arg)
                f_2[i] = (
                        -_b_eps * self.sigma * factors[4] * np.exp(_exp_arg)
                        / (self.distances - _sigma_cut)
                        / (self.distances - _sigma_cut)
                )

        potential_table = np.array(
            [v_1 + v_2, -1.0 * (f_1 + f_2)],
            dtype=np.float
        ).transpose()
        return potential_table
