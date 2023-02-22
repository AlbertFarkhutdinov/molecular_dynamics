from abc import abstractmethod

import numpy as np

from common.helpers import get_empty_vectors
from configurations import ThermodynamicSystem
from logs import LoggedObject


class BaseInitializer(LoggedObject):

    def __init__(self, initial_temperature: float = None) -> None:
        self.initial_temperature = initial_temperature

    @staticmethod
    def _get_initial_velocities(
            system: ThermodynamicSystem,
            temperature: float,
    ) -> ThermodynamicSystem:
        np.random.seed(0)
        _sigma = np.sqrt(temperature)
        _particles_number = system.particles_number
        velocities = get_empty_vectors(_particles_number).transpose()
        for i in range(3):
            velocities[i] = np.random.normal(
                0.0,
                _sigma,
                _particles_number,
            )
            velocities[i] -= velocities[i].sum() / _particles_number

        scale_factor = (
            np.sqrt(
                3.0 * temperature * _particles_number
                / (velocities * velocities).sum()
            )
        )
        for i in range(3):
            velocities[i] *= scale_factor
        system.velocities = velocities.transpose()
        return system

    @abstractmethod
    def _get_thermodynamic_system(self) -> ThermodynamicSystem:
        raise NotImplementedError('Define `_get_thermodynamic_system`.')

    def initialize(self) -> ThermodynamicSystem:
        system = self._get_thermodynamic_system()
        temperature = self.initial_temperature or system.temperature
        if temperature and system.velocities.sum() == 0.0:
            system = self._get_initial_velocities(
                system=system,
                temperature=temperature,
            )
        return system
