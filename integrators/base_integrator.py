from abc import ABC, abstractmethod
from typing import Optional

from configurations import ThermodynamicSystem
from core.external_parameters import ExternalParameters
from logs import LoggedObject


class BaseIntegrator(LoggedObject, ABC):

    def __init__(
            self,
            system: ThermodynamicSystem,
            time_step: float,
            external: Optional[ExternalParameters] = None,
    ) -> None:
        self.logger.debug(
            f'{self.__class__.__name__} instance initialization.'
        )
        self.system = system
        self.time_step = time_step
        self.external = external
        self.initial_temperature = self.system.temperature

    @abstractmethod
    def stage_1(self):
        raise NotImplementedError(
            'Define `stage_1` in'
            f'{self.__class__.__name__}.'
        )

    @abstractmethod
    def stage_2(self):
        raise NotImplementedError(
            'Define `stage_2` in'
            f'{self.__class__.__name__}.'
        )

    def after_stage(self, stage_id: int):
        kinetic_energy = self.system.kinetic_energy
        temperature = self.system.temperature
        volume = self.system.volume
        density = self.system.density
        pressure = self.system.pressure
        potential_energy = self.system.potential_energy
        total_energy = kinetic_energy + potential_energy
        log_postfix = f'after {self.__class__.__name__}.stage_{stage_id}()'
        self.logger.debug(f'Kinetic Energy {log_postfix}: {kinetic_energy};')
        self.logger.debug(f'Temperature {log_postfix}: {temperature};')
        self.logger.debug(f'Volume {log_postfix}: {volume};')
        self.logger.debug(f'Density {log_postfix}: {density};')
        self.logger.debug(f'Pressure {log_postfix}: {pressure};')
        self.logger.debug(f'Potential energy {log_postfix}: {potential_energy};')
        self.logger.debug(f'Total energy {log_postfix}: {total_energy};')
        if stage_id == 1:
            return kinetic_energy, temperature
        if stage_id == 2:
            return volume, density, pressure, total_energy
        return None

    def get_next_positions(
            self,
            acc_factor: float = 0.0,
    ) -> None:
        self.system.positions += (
                self.system.velocities * self.time_step
                + (
                        self.system.accelerations
                        - acc_factor * self.system.velocities
                )
                * (self.time_step * self.time_step) / 2.0
        )

    def get_next_velocities(
            self,
            vel_factor: float = 1.0,
            acc_factor: float = 0.0,
    ) -> None:
        self.system.velocities = (
                vel_factor * self.system.velocities
                + (
                        self.system.accelerations
                        - acc_factor * self.system.velocities
                )
                * self.time_step / 2.0
        )
