from abc import ABC, abstractmethod
from typing import Optional


from scripts_new.log_config import logger_wraps, log_debug_info
from scripts_new.system import System
from scripts_new.external_parameters import ExternalParameters


class BaseIntegrator(ABC):

    def __init__(
            self,
            system: System,
            time_step: float,
            external: Optional[ExternalParameters] = None,
    ) -> None:
        self.system = system
        self.time_step = time_step
        self.external = external
        self.initial_temperature = self.system.configuration.temperature

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
        kinetic_energy = self.system.configuration.kinetic_energy
        temperature = self.system.configuration.get_temperature(
            kinetic_energy=kinetic_energy,
        )
        volume = self.system.volume
        density = self.system.get_density(volume=volume)
        pressure = self.system.get_pressure(
            temperature=temperature,
            volume=volume,
            density=density,
        )
        potential_energy = self.system.potential_energy
        total_energy = kinetic_energy + potential_energy
        log_postfix = f'after {self.__class__.__name__}.stage_{stage_id}()'
        log_debug_info(f'Kinetic Energy {log_postfix}: {kinetic_energy};')
        log_debug_info(f'Temperature {log_postfix}: {temperature};')
        log_debug_info(f'Volume {log_postfix}: {volume};')
        log_debug_info(f'Density {log_postfix}: {density};')
        log_debug_info(f'Pressure {log_postfix}: {pressure};')
        log_debug_info(f'Potential energy {log_postfix}: {potential_energy};')
        log_debug_info(f'Total energy {log_postfix}: {total_energy};')
        if stage_id == 1:
            return kinetic_energy, temperature
        if stage_id == 2:
            return volume, density, pressure, total_energy
        return None

    @logger_wraps()
    def get_next_positions(
            self,
            acc_factor: float = 0.0,
    ) -> None:
        self.system.configuration.positions += (
                self.system.configuration.velocities * self.time_step
                + (
                        self.system.configuration.accelerations
                        - acc_factor * self.system.configuration.velocities
                )
                * (self.time_step * self.time_step) / 2.0
        )

    @logger_wraps()
    def get_next_velocities(
            self,
            vel_factor: float = 1.0,
            acc_factor: float = 0.0,
    ) -> None:
        self.system.configuration.velocities = (
                vel_factor * self.system.configuration.velocities
                + (
                        self.system.configuration.accelerations
                        - acc_factor * self.system.configuration.velocities
                )
                * self.time_step / 2.0
        )
