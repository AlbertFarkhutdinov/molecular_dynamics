import numpy as np

from helpers import get_empty_float_scalars


class PotentialParameters:

    def __init__(
            self,
            skin: float,
            potential_type: str,
    ):
        self.skin = skin
        self.update_test = True
        self.r_cut = 1.0
        if potential_type == 'lennard_jones':
            self.potential_table = self.load_lennard_jones_forces()
            self.r_cut *= 2.5
        elif potential_type == 'dzugutov':
            self.potential_table = self.load_dzugutov_forces()
            self.r_cut *= 1.94

    def load_lennard_jones_forces(self):
        sigma_pow6, epsilon_x4, r_cut = 1.0, 4.0, self.r_cut
        table_size = 25000
        r_ij = 0.0001 * np.arange(table_size) + 0.5
        energy_r_cut = epsilon_x4 * (sigma_pow6 ** 2 / (r_cut ** 12) - sigma_pow6 / (r_cut ** 6))
        energies = epsilon_x4 * (sigma_pow6 ** 2 / (r_ij ** 12) - sigma_pow6 / (r_ij ** 6)) - energy_r_cut
        forces = 6.0 * epsilon_x4 * (2.0 * sigma_pow6 ** 2 / (r_ij ** 13) - sigma_pow6 / (r_ij ** 7))
        potential_table = np.array([energies, forces], dtype=np.float).transpose()
        return potential_table

    def load_dzugutov_forces(self):
        sigma, epsilon, r_cut = 1.0, 1.0, self.r_cut
        table_size = 15000
        r_ij = 0.0001 * np.arange(table_size) + 0.5
        aa, bb, a, c, d, m = 5.82, 1.28, 1.87, 1.1, 0.27, 16
        v_1 = get_empty_float_scalars(table_size)
        v_2 = get_empty_float_scalars(table_size)
        f_1 = get_empty_float_scalars(table_size)
        f_2 = get_empty_float_scalars(table_size)
        for i in range(table_size):
            if r_ij[i] / sigma < a:
                v_1[i] = aa * epsilon * ((sigma / r_ij) ** m - bb) * np.exp(sigma * c / (r_ij - sigma * a))
                f_1[i] = (aa * epsilon * (-(m / sigma) * (sigma / r_ij) ** (m + 1) -
                                          (sigma / r_ij) ** m - bb) * sigma * c * (r_ij - sigma * a) ** (-2.0) *
                          np.exp(sigma * c / (r_ij - sigma * a)))
            if r_ij[i] / sigma < r_cut:
                v_2[i] = bb * epsilon * np.exp(sigma * r_cut / (r_ij - sigma * r_cut))
                f_2[i] = (-epsilon * sigma * d * bb *
                          np.exp(sigma * r_cut / (r_ij - sigma * r_cut)) *
                          (r_ij - sigma * r_cut) ** (-2.0))

        potential_table = np.array([v_1 + v_2, -1.0 * (f_1 + f_2)], dtype=np.float).transpose()
        return potential_table
