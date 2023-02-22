import pandas as pd

from common.constants import DATA_DIR
from configurations import PDBConverter, Snapshot, ThermodynamicSystem
from initializers.base_initializer import BaseInitializer


class FromFile(BaseInitializer):

    def __init__(
            self,
            file_name: str,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.file_name = str(DATA_DIR / file_name)

    def _get_thermodynamic_system(self) -> ThermodynamicSystem:
        velocities, accelerations = None, None
        if self.file_name.endswith('.pdb'):
            snapshot = PDBConverter().get_snapshot(self.file_name)
        elif self.file_name.endswith('.csv'):
            data = pd.read_csv(self.file_name, sep=';')
            cell_dimensions = data.loc[0, ['L_x', 'L_y', 'L_z']].to_numpy()
            positions = data[['x', 'y', 'z']].to_numpy()
            snapshot = Snapshot(
                positions=positions,
                cell_dimensions=cell_dimensions,
                is_pbc_applied=True,
                time=data.loc[0, 'time'],
            )
            velocities = data[['v_x', 'v_y', 'v_z']].to_numpy()
            accelerations = data[['a_x', 'a_y', 'a_z']].to_numpy()
        else:
            raise ValueError('Unacceptable file format')
        system = ThermodynamicSystem(
            positions=snapshot.positions,
            cell_dimensions=snapshot.cell_dimensions,
            is_pbc_applied=snapshot.is_pbc_applied,
            time=snapshot.time,
            velocities=velocities,
            accelerations=accelerations,
        )
        return system
