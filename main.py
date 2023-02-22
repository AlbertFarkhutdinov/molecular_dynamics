from typing import Dict, List

import numpy as np

from scripts.core import MolecularDynamics
from scripts.helpers import get_json, save_config_parameters
from scripts.log_config import logger_wraps


@logger_wraps()
def main(
        config_filenames: List[Dict[str, str]],
        is_with_isotherms: bool = True,
):
    _config_filename = config_filenames[0]
    md_instance = MolecularDynamics(
        config_filenames=_config_filename,
        is_with_isotherms=is_with_isotherms
    )
    md_instance.run_md()
    for i, file_name in enumerate(config_filenames[1:]):
        config_parameters = get_json(file_name)
        md_instance.update_simulation_parameters(config_parameters)
        md_instance.run_md()
        save_config_parameters(
            config_parameters=config_parameters,
            config_number=i + 1,
        )


if __name__ == '__main__':
    import os
    print(os.getcwd())
    np.set_printoptions(threshold=5000)
    IMMUTABLES = 'lennard_jones.json'
    SIMULATION_1 = {
        'immutables': IMMUTABLES,
        'initials': 'from_file_13e-1.json',
        'externals': 'mtk_nvt.json',
        'simulation_parameters': (
            'n005e3_conf_020e0_001e3_iso_005e3_eq_015e3_ens_005e3.json'
        ),
    }
    main(
        config_filenames=[
            SIMULATION_1,
        ],
        is_with_isotherms=False,
    )
