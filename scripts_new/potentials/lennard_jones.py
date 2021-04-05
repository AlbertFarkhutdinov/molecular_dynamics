import numpy as np

from scripts_new.potentials.base_potential import BasePotential


class LennardJones(BasePotential):

    def __repr__(self):
        return f'{self.__class__.__name__}(table_size = {self.table_size!r})'

    def get_energies_and_forces(self):
        # forces - это величины -dU/rdr.
        # Сила получается умножением на вектор r_ij
        sigma_pow6, epsilon_x4 = self.sigma ** 6, self.epsilon * 4
        energy_r_cut = epsilon_x4 * (
                sigma_pow6 * sigma_pow6 / (self.r_cut ** 12)
                - sigma_pow6 / (self.r_cut ** 6)
        )
        energies = epsilon_x4 * (
                sigma_pow6 * sigma_pow6 / (self.distances ** 12)
                - sigma_pow6 / (self.distances ** 6)
        ) - energy_r_cut
        derivatives = 6.0 * epsilon_x4 * (
                2.0 * sigma_pow6 * sigma_pow6 / (self.distances ** 13)
                - sigma_pow6 / (self.distances ** 7)
        )
        forces = derivatives / self.distances
        potential_table = np.array(
            [energies, forces],
            dtype=np.float,
        ).transpose()
        return potential_table
