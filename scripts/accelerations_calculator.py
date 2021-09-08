import numpy as np

from scripts.immutable_parameters import ImmutableParameters
from scripts.system import System
from scripts.helpers import get_empty_float_scalars, get_empty_int_scalars
from scripts.helpers import get_empty_vectors
from scripts.log_config import log_debug_info, logger_wraps
from scripts.numba_procedures import lf_cycle, update_list_cycle


class AccelerationsCalculator:

    def __init__(
            self,
            immutables: ImmutableParameters,
            system: System,
    ):
        log_debug_info(f'{self.__class__.__name__} instance initialization.')
        self.immutables = immutables
        log_debug_info(f'Immutables = {self.immutables}')
        self.system = system
        self.neighbours_lists = {
            'all_neighbours': get_empty_int_scalars(
                self.immutables.particles_number
            ),
            'first_neighbours': get_empty_int_scalars(
                self.immutables.particles_number
            ),
            'last_neighbours': get_empty_int_scalars(
                100 * self.immutables.particles_number
            ),
        }
        self.update_test = True
        self.displacements = get_empty_vectors(
            self.immutables.particles_number
        )

    @logger_wraps()
    def load_forces(self):
        potential_energies = get_empty_float_scalars(
            self.immutables.particles_number
        )
        self.system.configuration.accelerations = get_empty_vectors(
            self.immutables.particles_number
        )
        if self.update_test:
            log_debug_info('update_test = True')
            self.update_list()
            self.displacements = get_empty_vectors(
                self.immutables.particles_number
            )
            self.update_test = False

        log_debug_info(
            f'particles_number = {self.immutables.particles_number}'
        )
        log_debug_info(f'r_cut = {self.immutables.r_cut}')
        log_debug_info(f'cell_dimensions = {self.system.cell_dimensions}')
        log_debug_info(
            f'all_neighbours.max() = {np.max(self.neighbours_lists["all_neighbours"])}'
        )
        log_debug_info(
            f'first_neighbours.max() = {np.max(self.neighbours_lists["first_neighbours"])}'
        )
        log_debug_info(
            f'last_neighbours.max() = {np.max(self.neighbours_lists["last_neighbours"])}'
        )
        log_debug_info(
            f'positions.mean() = {self.system.configuration.positions.mean()}'
        )
        log_debug_info(
            f'accelerations.mean() = {self.system.configuration.accelerations.mean()}'
        )
        log_debug_info(
            f'potential_table[:, 0].min() = {self.immutables.potential_table[:, 0].min()}'
        )
        log_debug_info(
            f'potential_table[:, 0].mean() = {self.immutables.potential_table[:, 0].mean()}'
        )
        log_debug_info(
            f'potential_table[:, 0].max() = {self.immutables.potential_table[:, 0].max()}'
        )
        log_debug_info(
            f'potential_table[:, 1].min() = {self.immutables.potential_table[:, 1].min()}'
        )
        log_debug_info(
            f'potential_table[:, 1].mean() = {self.immutables.potential_table[:, 1].mean()}'
        )
        log_debug_info(
            f'potential_table[:, 1].max() = {self.immutables.potential_table[:, 1].max()}'
        )
        log_debug_info(f'Virial = {self.system.virial}')
        log_debug_info(f'Mean potential: {potential_energies.mean()};')
        log_debug_info(f'Potential energy: {potential_energies.sum()};')
        self.system.virial = lf_cycle(
            particles_number=self.immutables.particles_number,
            r_cut=self.immutables.r_cut,
            cell_dimensions=self.system.cell_dimensions,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
            potential_table=self.immutables.potential_table,
            potential_energies=potential_energies,
            positions=self.system.configuration.positions,
            accelerations=self.system.configuration.accelerations,
        )
        log_debug_info(f'Virial = {self.system.virial}')
        log_debug_info(f'Mean potential: {potential_energies.mean()};')
        log_debug_info(f'Potential energy: {potential_energies.sum()};')
        acc_mag = (
                          self.system.configuration.accelerations ** 2
                  ).sum(axis=1) ** 0.5
        log_debug_info(
            f'Mean and max acceleration: {acc_mag.mean()}, {acc_mag.max()}'
        )
        potential_energy = potential_energies.sum()
        self.displacements += (
                self.system.configuration.velocities
                * self.immutables.time_step
                + self.system.configuration.accelerations
                * self.immutables.time_step
                * self.immutables.time_step / 2.0
        )
        self.load_move_test()
        self.system.potential_energy = potential_energy

    @logger_wraps()
    def update_list(self):
        self.neighbours_lists = {
            'first_neighbours': get_empty_int_scalars(
                self.immutables.particles_number
            ),
            'last_neighbours': get_empty_int_scalars(
                self.immutables.particles_number
            ),
            'all_neighbours': get_empty_int_scalars(
                100 * self.immutables.particles_number
            ),
        }
        advances = get_empty_int_scalars(self.immutables.particles_number)
        update_list_cycle(
            rng=self.immutables.r_cut + self.immutables.skin_for_neighbors,
            advances=advances,
            particles_number=self.immutables.particles_number,
            positions=self.system.configuration.positions,
            cell_dimensions=self.system.cell_dimensions,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
        )

    @logger_wraps()
    def load_move_test(self):
        ds_1, ds_2 = 0.0, 0.0
        for i in range(self.immutables.particles_number):
            _ds = (self.displacements[i] ** 2).sum() ** 0.5
            if _ds >= ds_1:
                ds_2 = ds_1
                ds_1 = _ds
            elif _ds >= ds_2:
                ds_2 = _ds
        self.update_test = (ds_1 + ds_2) > self.immutables.skin_for_neighbors
