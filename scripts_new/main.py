"""
Liquid argon:
_____________
mass = 6.69E-26 kilograms
distance = sigma = 0.341E-9 meters
energy = epsilon = 1.65E-21 joules
temperature = epsilon / k_B = 119.8 kelvin
tau = sigma * sqrt(mass / epsilon) = 2.17E-12 seconds
velocity = sigma / tau = 1.57E2 m/s
force = epsilon/sigma = 4.85E-12 newtons
pressure = epsilon / (sigma ^ 3) = 4.2E7 pascal = 4.2E2 atmospheres
heating velocity = epsilon / k_B / tau = 0.55E14 K/s

"""

from typing import List

import numpy as np

from scripts_new.core import MolecularDynamics
from scripts_new.helpers import get_config_parameters


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
    for file_name in config_filenames[1:]:
        config_parameters = get_config_parameters(file_name)
        md.update_simulation_parameters(config_parameters)
        md.run_md()

    # TODO postprocessor


if __name__ == '__main__':
    # TODO check potential at T = 2.8 (compare 2020-12-17 and the book, p.87)
    np.set_printoptions(threshold=5000)

    # CONFIG_FILE_NAME = 'book_chapter_4_stage_1.json'
    CONFIG_FILE_NAME = 'test_2.json'
    # CONFIG_FILE_NAME = 'cooling.json'
    # CONFIG_FILE_NAME = 'book_chapter_4_stage_2.json'
    # CONFIG_FILE_NAME = 'slow_cooling.json'
    # CONFIG_FILE_NAME = 'npt_2.8.json'
    # CONFIG_FILE_NAME = 'equilibrium_2.8.json'
    # CONFIG_FILE_NAME = 'equilibrium_0.01.json'
    # CONFIG_FILE_NAME = 'calculation_time_test.json'

    main(
        config_filenames=[
            'test_1.json',
            'test_2.json',
            'test_3.json',
        ],
        is_with_isotherms=True,
    )
