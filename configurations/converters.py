from abc import ABC, abstractmethod

import numpy as np
from pretty_repr import RepresentableObject

from configurations import Snapshot


class BaseConverter(ABC, RepresentableObject):

    @abstractmethod
    def get_string(self, snapshot: Snapshot, **kwargs) -> str:
        raise NotImplementedError

    def get_snapshot(self, file_name: str) -> Snapshot:
        pass


class LMPConverter(BaseConverter):

    def get_string(self, snapshot: Snapshot, **kwargs) -> str:
        lines = [
            'ITEM: TIMESTEP',
            str(f'{snapshot.time:.5f}'),
            'ITEM: NUMBER OF ATOMS',
            str(snapshot.particles_number),
            'ITEM: BOX BOUNDS pp pp pp',
            *(
                f'{-dim / 2} {dim / 2}'
                for dim in snapshot.cell_dimensions
            ),
            'ITEM: ATOMS id type x y z',
            *(
                f'{i + 1} 0 {pos[0]} {pos[1]} {pos[2]}'
                for i, pos in enumerate(snapshot.positions)
            ),
            '\n',
        ]
        return '\n'.join(lines)


class XYZConverter(BaseConverter):

    def get_string(self, snapshot: Snapshot, **kwargs) -> str:
        _step = kwargs.get('step', 1)
        cell_dimensions_string = ','.join(
            snapshot.cell_dimensions.astype(str)
        )
        lines = [
            f'{snapshot.particles_number}',
            f'step: {_step} columns: name, pos cell:{cell_dimensions_string}',
        ]
        for position in snapshot.positions:
            lines.append(
                'A' + ''.join([f'{position[i]:15.6f}' for i in range(3)])
            )
        return '\n'.join(lines)


class PDBConverter(BaseConverter):

    def get_string(self, snapshot: Snapshot, **kwargs) -> str:
        lines = []
        for i, position in enumerate(snapshot.positions):
            line = f'ATOM  {str(i + 1).rjust(5)} '
            _atom_name = 'Ar'
            line += _atom_name.ljust(5)
            line += 'MOL '
            line += ' '
            line += str(i + 1).rjust(4)
            line += '    '
            for component in position:
                line += f'{component:8.3f}'
            _occupancy = 1.00
            _temp_factor = 0.00
            line += (
                f'{_occupancy:6.2f}{_temp_factor:6.2f}          {_atom_name} 0'
            )
            lines.append(line)
        return '\n'.join(lines)

    def get_snapshot(self, file_name: str) -> Snapshot:
        positions = []
        cell_dimensions = []
        with open(file_name, mode='r', encoding='utf8') as file:
            for line in file:
                if 'Boundary Conditions' in line:
                    _items = line.split()
                    cell_dimensions = np.array([
                        _items[-9], _items[-5], _items[-1]
                    ]).astype(float)
                elif 'ATOM' in line:
                    positions.append(np.array([
                        line[30:38].strip(),
                        line[38:46].strip(),
                        line[46:54].strip(),
                    ]).astype(float))
        positions = np.array(positions)
        return Snapshot(
            positions=positions,
            cell_dimensions=cell_dimensions,
            is_pbc_applied=False,
        )
