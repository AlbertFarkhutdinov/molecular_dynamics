import numpy as np


class PrincipalComponentAnalysis:

    def __init__(
            self,
            iterations_number: int,
            fitness: callable,
            parameters_number: int,
            bounds: tuple[float, float]
    ):
        self.iterations_number = iterations_number
        self.fitness = fitness
        self.parameters_number = parameters_number
        self.inferior_limits = bounds[0] * np.ones(self.parameters_number)
        self.superior_limits = bounds[1] * np.ones(self.parameters_number)
        self.old_config = self.get_random_configuration()
        self.new_config = np.zeros(self.parameters_number)
        self.best_fitness = self.fitness(self.old_config)

    def get_random_configuration(self):
        return (
                self.inferior_limits
                + (self.superior_limits - self.inferior_limits)
                * np.random.random(self.parameters_number)
        )

    def perturbation(self):
        print('Perturbation.')
        rand = np.random.random(self.parameters_number)
        self.new_config = (
                self.old_config
                + ((self.superior_limits - self.old_config) * rand)
                - ((self.old_config - self.inferior_limits) * (1 - rand))
        )
        self.new_config = np.min(
            [self.new_config, self.superior_limits],
            axis=0,
        )
        self.new_config = np.max(
            [self.new_config, self.inferior_limits],
            axis=0,
        )

    def small_perturbation(self):
        # print('Small Perturbation.')
        uppers = np.min(
            [
                (1.0 + 0.2 * np.random.random(self.parameters_number))
                * self.old_config,
                self.superior_limits
            ],
            axis=0,
        )
        lowers = np.max(
            [
                (0.8 + 0.2 * np.random.random(self.parameters_number))
                * self.old_config,
                self.inferior_limits
            ],
            axis=0,
        )
        rand = np.random.random(self.parameters_number)
        self.new_config = (
                self.old_config
                + ((uppers - self.old_config) * rand)
                - ((self.old_config - lowers) * (1 - rand))
        )

    def exploration(self):
        print('Exploration.')
        for _ in range(self.iterations_number):
            self.small_perturbation()
            if self.fitness(self.new_config) > self.fitness(self.old_config):
                if self.fitness(self.new_config) > self.best_fitness:
                    self.best_fitness = self.fitness(self.new_config)
                self.old_config = self.new_config

    def scattering(self):
        print('Scattering.')
        p_scattering = 1 - self.fitness(self.new_config) / self.best_fitness
        if p_scattering > np.random.random():
            self.old_config = self.get_random_configuration()
        else:
            self.exploration()

    def run_pca_step(self):
        self.perturbation()
        if self.fitness(self.new_config) > self.fitness(self.old_config):
            if self.fitness(self.new_config) > self.best_fitness:
                self.best_fitness = self.fitness(self.new_config)
            self.old_config = self.new_config
            self.exploration()
        else:
            self.scattering()

    def run_pca(self):
        for i in range(self.iterations_number):
            print(f'Iteration {i + 1} / {self.iterations_number}.')
            self.run_pca_step()
        return self.new_config
