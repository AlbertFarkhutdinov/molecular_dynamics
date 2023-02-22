from configurations.dynamic_system import DynamicSystem


class ThermodynamicSystem(DynamicSystem):

    @property
    def temperature(self) -> float:
        __temperature = 2.0 * self.kinetic_energy / 3.0 / self.particles_number
        self.logger.debug(f'Temperature = {__temperature}')
        return __temperature

    @property
    def pressure(self) -> float:
        __pressure = (
                self.density * self.temperature + self.vir / (3 * self.volume)
        )
        self.logger.debug(f'Pressure = {__pressure}')
        return __pressure
