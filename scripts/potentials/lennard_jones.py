import numpy as np

from scripts.potentials.base_potential import BasePotential


class LennardJones(BasePotential):

    def __init__(self):
        super().__init__()
        self.table_size = 25000
        self.distances = 0.5 + 0.0001 * np.arange(1, self.table_size + 1)
        self.r_cut = 2.5
        self.potential_table = self.get_energies_and_forces()

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
