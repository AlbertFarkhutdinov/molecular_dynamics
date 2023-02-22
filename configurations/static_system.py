import numpy as np

from common.numba_procedures import get_boundary_conditions
from logs import LoggedObject


class StaticSystem(LoggedObject):

    def __init__(
            self,
            positions: np.ndarray,
            cell_dimensions: np.ndarray,
            is_pbc_applied: bool,
    ) -> None:
        self.is_pbc_applied = is_pbc_applied
        self.cell_dimensions = cell_dimensions
        self.positions = positions
        self.annul_system_center()
        if is_pbc_applied:
            self.apply_boundary_conditions()

    @property
    def is_pbc_applied(self) -> bool:
        return self.__is_pbc_applied

    @is_pbc_applied.setter
    def is_pbc_applied(self, is_pbc_applied: bool) -> None:
        self.__is_pbc_applied = is_pbc_applied

    @property
    def positions(self) -> np.ndarray:
        return self.__positions

    @positions.setter
    def positions(self, positions: np.ndarray) -> None:
        if not (len(positions.shape) == 2 and positions.shape[1] == 3):
            raise ValueError('Unacceptable positions.')
        self.__positions = positions
        if self.is_pbc_applied:
            self.apply_boundary_conditions()

    @property
    def cell_dimensions(self) -> np.ndarray:
        return self.__cell_dimensions

    @cell_dimensions.setter
    def cell_dimensions(self, cell_dimensions: np.ndarray) -> None:
        if cell_dimensions.shape != (3,):
            raise ValueError('Unacceptable cell dimensions.')
        self.__cell_dimensions = cell_dimensions

    @property
    def particles_number(self) -> int:
        return self.positions.shape[0]

    @property
    def volume(self) -> float:
        volume = self.cell_dimensions.prod()
        self.logger.debug(f'Volume = {volume}')
        return volume

    @property
    def density(self) -> float:
        density = self.particles_number / self.volume
        self.logger.debug(f'Density = {density}')
        return density

    def get_system_center(self):
        return self.positions.sum(axis=0) / self.particles_number

    def annul_system_center(self) -> None:
        self.logger.debug('System center is set to zero.')
        self.positions -= self.get_system_center()

    def apply_boundary_conditions(self) -> None:
        self.logger.debug('apply_boundary_conditions')
        self.logger.debug(f'positions.mean() = {self.positions.mean()}')
        self.positions = get_boundary_conditions(
            cell_dimensions=self.cell_dimensions,
            particles_number=self.particles_number,
            positions=self.positions,
        )
        self.logger.debug(f'positions.mean() = {self.positions.mean()}')
