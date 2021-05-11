import numpy as np

from scripts.potentials.base_potential import BasePotential
from scripts.log_config import logger_wraps, log_debug_info


class LennardJones(BasePotential):

    def __repr__(self):
        return f'{self.__class__.__name__}(table_size = {self.table_size!r})'

    @logger_wraps()
    def get_energies_and_forces(self):
        # forces - это величины -dU/rdr.
        # Сила получается умножением на вектор r_ij
        sigma_pow6, epsilon_x4, r_cut = self.sigma ** 6, self.epsilon * 4, 1.0#self.r_cut
        log_debug_info(
            'sigma_pow6, epsilon_x4, r_cut, table_size = '
            f'{sigma_pow6}, {epsilon_x4}, {self.r_cut}, {self.table_size}'
        )
        log_debug_info(
            f'r_ij = {self.distances.min()}, {self.distances.mean()}, '
            f'{self.distances.max()}'
        )
        energy_r_cut = epsilon_x4 * (
                sigma_pow6 * sigma_pow6 / (r_cut ** 12)
                - sigma_pow6 / (r_cut ** 6)
        )
        log_debug_info(f'energy_r_cut = {energy_r_cut}')
        energies = epsilon_x4 * (
                sigma_pow6 * sigma_pow6 / (self.distances ** 12)
                - sigma_pow6 / (self.distances ** 6)
        ) - energy_r_cut
        log_debug_info(
            f'energies = {energies.min()}, {energies.mean()}, {energies.max()}'
        )

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
