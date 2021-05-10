import os
from typing import Optional

import numpy as np

from scripts.constants import PATH_TO_DATA
from scripts.helpers import get_date
from scripts.numba_procedures import get_boundary_conditions
from scripts.log_config import log_debug_info


class Configuration:

    def __init__(
            self,
            **configuration_kwargs,
    ):
        self.particles_number = configuration_kwargs.get('particles_number', 0)
        self.positions = configuration_kwargs.get('positions')
        self.velocities = configuration_kwargs.get('velocities')
        self.accelerations = configuration_kwargs.get('accelerations')
        self.temperature = configuration_kwargs.get('temperature', 0.0)

    @property
    def kinetic_energy(self) -> float:
        return (self.velocities * self.velocities).sum() / 2.0

    def get_temperature(
            self,
            kinetic_energy: Optional[float] = None,
    ) -> float:
        _kinetic_energy = kinetic_energy or self.kinetic_energy
        temperature = 2.0 * _kinetic_energy / 3.0 / self.particles_number
        self.temperature = temperature
        return temperature

    def get_system_center(self):
        return self.positions.sum(axis=0) / self.particles_number

    def get_msd(
            self,
            previous_positions
    ):
        return (
                ((self.positions - previous_positions) ** 2).sum()
                / self.particles_number
        )


class System:

    def __init__(
            self,
            **system_kwargs,
    ):
        self.time = system_kwargs.get('time', 0.0)
        self.configuration = system_kwargs.get(
            'configuration',
            Configuration(),
        )
        self.cell_dimensions = system_kwargs.get(
            'cell_dimensions',
            np.zeros(3, dtype=np.float),
        )
        self.density = system_kwargs.get('density', 0.0)
        self.virial = system_kwargs.get('virial', 0.0)
        self.pressure = system_kwargs.get('pressure', 0.0)
        self.potential_energy = system_kwargs.get('potential_energy', 0.0)

    @property
    def volume(self) -> float:
        return self.cell_dimensions.prod()

    def get_density(
            self,
            volume: float,
    ) -> float:
        density = self.configuration.particles_number / volume
        self.density = density
        return density

    def get_pressure(
            self,
            virial: Optional[float] = None,
            temperature: Optional[float] = None,
            density: Optional[float] = None,
            volume: Optional[float] = None,
    ) -> float:
        _density = density or self.density
        _temperature = temperature or self.configuration.temperature
        _virial = virial or self.virial
        _volume = volume or self.volume
        pressure = _density * _temperature + _virial / (3 * _volume)
        log_debug_info(f'_density = {_density}')
        log_debug_info(f'_temperature = {_temperature}')
        log_debug_info(f'_virial = {_virial}')
        log_debug_info(f'_volume = {_volume}')
        log_debug_info(f'pressure = {pressure}')
        self.pressure = pressure
        return pressure

    def apply_boundary_conditions(self, positions=None) -> None:
        log_debug_info(f'apply_boundary_conditions')
        _positions = (
            self.configuration.positions
            if positions is None
            else positions
        )
        log_debug_info(f'positions.min() = {_positions.min()}')
        log_debug_info(f'positions.mean() = {_positions.mean()}')
        log_debug_info(f'positions.max() = {_positions.max()}')
        self.configuration.positions = get_boundary_conditions(
            cell_dimensions=self.cell_dimensions,
            particles_number=self.configuration.particles_number,
            positions=_positions,
        )
        log_debug_info(f'positions.min() = {_positions.min()}')
        log_debug_info(f'positions.mean() = {_positions.mean()}')
        log_debug_info(f'positions.max() = {_positions.max()}')

    def save_xyz_file(self, filename: str, step: int):
        _path = os.path.join(
            os.path.join(
                PATH_TO_DATA,
                get_date(),
            ),
            filename,
        )
        _mode = 'a' if os.path.exists(_path) else 'w'
        with open(_path, mode=_mode, encoding='utf8') as file:
            file.write(
                f'{self.configuration.particles_number}\n')
            file.write(f'step: {step} columns: name, pos cell:')
            file.write(f"{','.join(self.cell_dimensions.astype(str))}\n")
            for position in self.configuration.positions:
                file.write('A')
                for i in range(3):
                    file.write(f'{position[i]:15.6f}')
                file.write('\n')
