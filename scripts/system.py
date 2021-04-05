from scripts.helpers import get_empty_vectors


class System:

    def __init__(
            self,
            particles_number: int,
    ):
        self.positions = get_empty_vectors(particles_number)
        self.velocities = get_empty_vectors(particles_number)
        self.accelerations = get_empty_vectors(particles_number)
        self.temperature = 0.0
        self.volume = 0.0
        self.virial = 0.0
        self.pressure = 0.0
        self.potential_energy = 0.0
