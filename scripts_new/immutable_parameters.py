from scripts_new.potentials.lennard_jones import LennardJones
from scripts_new.potentials.dzugutov import Dzugutov


class ImmutableParameters:

    def __init__(
            self,
            **immutables,
    ):
        self.particles_number = immutables.get('particles_number')
        self.time_step = immutables.get('time_step')
        _potential = self.get_potential(
            potential_type=immutables.get('potential_type'),
        )
        self.r_cut = _potential.r_cut
        self.potential_table = _potential.potential_table
        self.skin_for_neighbors = immutables.get('skin_for_neighbors')

    def __repr__(self):
        return ''.join([
            f'{self.__class__.__name__}(',
            f'particles_number={self.particles_number!r}',
            ', '
            f'time_step={self.time_step!r}',
            ', '
            f'r_cut={self.r_cut!r}',
            ', '
            f'skin_for_neighbors={self.skin_for_neighbors!r}',
            ')'
        ])

    @staticmethod
    def get_potential(potential_type: str):
        potential = None
        if potential_type == 'lennard_jones':
            potential = LennardJones(table_size=25000, r_cut=2.5)
        elif potential_type == 'dzugutov':
            potential = Dzugutov(table_size=15000, r_cut=1.94)
        return potential
