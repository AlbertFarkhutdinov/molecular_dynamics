from datetime import datetime

import numpy as np


np.random.seed(0)


class RadialMovementOptimization:

    def __init__(
            self,
            centre: np.ndarray,
            generations_number: int,
            particles_number: int,
            dimensions_number: int,
            bounds: tuple[float, float],
    ):
        self.generations_number = generations_number
        self.particles_number = particles_number
        self.dimensions_number = dimensions_number
        self.bounds = bounds
        self.locations = self.bounds[0] + np.random.uniform(
            size=(particles_number, dimensions_number)
        ) * (self.bounds[1] - self.bounds[0])
        self.centre = centre
        self.global_best = np.array(self.centre)

    def generate_velocities(self, denominator: int):
        return np.random.uniform(
            low=-1.0,
            size=(self.particles_number, self.dimensions_number)
        ) * (self.bounds[1] - self.bounds[0]) / denominator

    def get_weight(
            self,
            generation: int,
            weight_limits: tuple[float, float],
    ) -> float:
        return (
                weight_limits[1]
                - (weight_limits[1] - weight_limits[0])
                * generation / self.generations_number
        )

    def check_constraints(self, weight: float, velocities: np.ndarray):
        for i in range(self.particles_number):
            for j in range(self.dimensions_number):
                self.locations[i][j] = (
                        self.centre[j] + weight * velocities[i][j]
                )
                if self.locations[i][j] > self.bounds[1]:
                    self.locations[i][j] = self.bounds[1]
                if self.locations[i][j] < self.bounds[0]:
                    self.locations[i][j] = self.bounds[0]

    def optimize(
            self,
            func: callable,
            scale: int,
            c_parameters: tuple[float, float],
            weight_limits: tuple[float, float],
    ):
        global_best, radial_best = np.array(self.centre), np.array(self.centre)
        temp = np.array(self.centre)
        global_minimum = func(global_best)
        print(f'{global_minimum = :.5f}')
        for generation in range(self.generations_number):
            generation_time = datetime.now()
            generation_minimum = func(np.array(self.centre))
            # denominator = scale * (generation + 1)
            denominator = scale
            velocities = self.generate_velocities(denominator=denominator)
            weight = self.get_weight(
                generation=generation,
                weight_limits=weight_limits,
            )
            self.check_constraints(weight=weight, velocities=velocities)
            for _, location in enumerate(self.locations):
                out = func(location)
                if out < generation_minimum:
                    generation_minimum = out
                    radial_best = np.array(location)
                    if generation_minimum < global_minimum:
                        global_minimum = generation_minimum
                        temp = radial_best
            self.centre += c_parameters[0] * (global_best - self.centre)
            self.centre += c_parameters[1] * (radial_best - self.centre)
            self.global_best = temp
            print(
                f'gen = {generation}',
                f'gen_min = {generation_minimum:.4f}',
                f'global = {global_minimum:.4f}',
                f'denominator = {denominator}',
                f'execution_time = {datetime.now() - generation_time}',
                sep='; '
            )
        return self.global_best


if __name__ == '__main__':

    def de_jong_1(args):
        sum_ = 0
        for i, arg in enumerate(args):
            sum_ += abs(arg - i)
        return sum_


    G_BEST = RadialMovementOptimization(
        generations_number=10000,
        particles_number=10,
        dimensions_number=3,
        bounds=(-5.12, 5.12),
        centre=np.array([1.57, 2.18, 3.92])
    ).optimize(
        func=de_jong_1,
        c_parameters=(0.6, 0.7),
        weight_limits=(0, 1),
        scale=2,
    )
    print(G_BEST)
