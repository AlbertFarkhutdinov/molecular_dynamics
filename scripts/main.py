from typing import List

import numpy as np

from scripts.core import MolecularDynamics
from scripts.helpers import get_config_parameters, save_config_parameters
from scripts.log_config import logger_wraps


@logger_wraps()
def main(
        config_filenames: List[str],
        is_with_isotherms: bool = True,
):
    _config_filename = config_filenames[0]
    md = MolecularDynamics(
        config_filename=_config_filename,
        is_with_isotherms=is_with_isotherms
    )
    md.run_md()
    save_config_parameters(
        config_parameters=get_config_parameters(_config_filename),
        config_number=0,
    )
    for i, file_name in enumerate(config_filenames[1:]):
        config_parameters = get_config_parameters(file_name)
        md.update_simulation_parameters(config_parameters)
        md.run_md()
        save_config_parameters(
            config_parameters=config_parameters,
            config_number=i + 1,
        )


if __name__ == '__main__':
    np.set_printoptions(threshold=5000)
    # TODO NPT
    main(
        config_filenames=[
            'test_npt.json',
        ],
        is_with_isotherms=True,
    )
