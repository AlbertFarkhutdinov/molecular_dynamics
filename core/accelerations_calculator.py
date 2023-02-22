import numpy as np

from common.helpers import get_empty_float_scalars, get_empty_int_scalars
from common.helpers import get_empty_vectors
from common.numba_procedures import lf_cycle, update_list_cycle
from configurations import DynamicSystem
from core.immutable_parameters import ImmutableParameters
from logs import LoggedObject


class AccelerationsCalculator(LoggedObject):

    def __init__(
            self,
            immutables: ImmutableParameters,
            system: DynamicSystem,
    ):
        self.logger.debug(
            f'{self.__class__.__name__} instance initialization.'
        )
        self.immutables = immutables
        self.logger.debug(f'Immutables = {self.immutables}')
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

    def load_forces(self):
        potential_energies = get_empty_float_scalars(
            self.immutables.particles_number
        )
        self.system.accelerations = get_empty_vectors(
            self.immutables.particles_number
        )
        if self.update_test:
            self.logger.debug('update_test = True')
            self.update_list()
            self.displacements = get_empty_vectors(
                self.immutables.particles_number
            )
            self.update_test = False

        self.logger.debug(
            f'particles_number = {self.immutables.particles_number}'
        )
        self.logger.debug(f'r_cut = {self.immutables.r_cut}')
        self.logger.debug(f'cell_dimensions = {self.system.cell_dimensions}')
        self.logger.debug(
            f'all_neighbours.max() = {np.max(self.neighbours_lists["all_neighbours"])}'
        )
        self.logger.debug(
            f'first_neighbours.max() = {np.max(self.neighbours_lists["first_neighbours"])}'
        )
        self.logger.debug(
            f'last_neighbours.max() = {np.max(self.neighbours_lists["last_neighbours"])}'
        )
        self.logger.debug(
            f'positions.mean() = {self.system.positions.mean()}'
        )
        self.logger.debug(
            f'accelerations.mean() = {self.system.accelerations.mean()}'
        )
        self.logger.debug(
            f'potential_table[:, 0].min() = {self.immutables.potential_table[:, 0].min()}'
        )
        self.logger.debug(
            f'potential_table[:, 0].mean() = {self.immutables.potential_table[:, 0].mean()}'
        )
        self.logger.debug(
            f'potential_table[:, 0].max() = {self.immutables.potential_table[:, 0].max()}'
        )
        self.logger.debug(
            f'potential_table[:, 1].min() = {self.immutables.potential_table[:, 1].min()}'
        )
        self.logger.debug(
            f'potential_table[:, 1].mean() = {self.immutables.potential_table[:, 1].mean()}'
        )
        self.logger.debug(
            f'potential_table[:, 1].max() = {self.immutables.potential_table[:, 1].max()}'
        )
        self.logger.debug(f'Vir = {self.system.vir}')
        self.logger.debug(f'Mean potential: {potential_energies.mean()};')
        self.logger.debug(f'Potential energy: {potential_energies.sum()};')
        self.system.vir = lf_cycle(
            particles_number=self.immutables.particles_number,
            r_cut=self.immutables.r_cut,
            cell_dimensions=self.system.cell_dimensions,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
            potential_table=self.immutables.potential_table,
            potential_energies=potential_energies,
            positions=self.system.positions,
            accelerations=self.system.accelerations,
        )
        self.logger.debug(f'Vir = {self.system.vir}')
        self.logger.debug(f'Mean potential: {potential_energies.mean()};')
        self.logger.debug(f'Potential energy: {potential_energies.sum()};')
        acc_mag = (
                          self.system.accelerations ** 2
                  ).sum(axis=1) ** 0.5
        self.logger.debug(
            f'Mean and max acceleration: {acc_mag.mean()}, {acc_mag.max()}'
        )
        potential_energy = potential_energies.sum()
        self.displacements += (
                self.system.velocities
                * self.immutables.time_step
                + self.system.accelerations
                * self.immutables.time_step
                * self.immutables.time_step / 2.0
        )
        self.load_move_test()
        self.system.potential_energy = potential_energy

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
            positions=self.system.positions,
            cell_dimensions=self.system.cell_dimensions,
            all_neighbours=self.neighbours_lists['all_neighbours'],
            first_neighbours=self.neighbours_lists['first_neighbours'],
            last_neighbours=self.neighbours_lists['last_neighbours'],
        )

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
