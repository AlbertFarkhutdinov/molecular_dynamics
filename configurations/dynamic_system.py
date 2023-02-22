import numpy as np

from common.helpers import get_empty_vectors
from configurations.snapshot import Snapshot


class DynamicSystem(Snapshot):

    def __init__(
            self,
            velocities: np.ndarray = None,
            accelerations: np.ndarray = None,
            vir: float = 0.0,
            potential_energy: float = 0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.velocities = (
            velocities
            if velocities is None
            else get_empty_vectors(self.particles_number)
        )
        self.accelerations = (
            accelerations
            if accelerations is None
            else get_empty_vectors(self.particles_number)
        )
        self.vir = vir
        self.potential_energy = potential_energy

    @property
    def velocities(self) -> np.ndarray:
        return self.__velocities

    @velocities.setter
    def velocities(self, velocities: np.ndarray) -> None:
        if not (len(velocities.shape) == 2 and velocities.shape[1] == 3):
            raise ValueError('Unacceptable velocities.')
        self.__velocities = velocities

    @property
    def accelerations(self) -> np.ndarray:
        return self.__accelerations

    @accelerations.setter
    def accelerations(self, accelerations: np.ndarray) -> None:
        if not (len(accelerations.shape) == 2 and accelerations.shape[1] == 3):
            raise ValueError('Unacceptable accelerations.')
        self.__accelerations = accelerations

    @property
    def vir(self) -> float:
        return self.__vir

    @vir.setter
    def vir(self, vir: float) -> None:
        self.__vir = vir

    @property
    def potential_energy(self) -> float:
        return self.__potential_energy

    @potential_energy.setter
    def potential_energy(self, potential_energy: float) -> None:
        self.__potential_energy = potential_energy

    @property
    def kinetic_energy(self) -> float:
        __kinetic_energy = (self.velocities * self.velocities).sum() / 2.0
        self.logger.debug(f'Kinetic Energy = {__kinetic_energy}')
        return __kinetic_energy
