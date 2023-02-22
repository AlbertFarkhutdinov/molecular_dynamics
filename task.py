import os

from main import main
from common.helpers import get_date


IMMUTABLES = 'lennard_jones.json'
SIMULATION_1 = {
    'immutables': IMMUTABLES,
    'initials': 'from_file_13e-1.json',
    'externals': 'velocity_scaling_HV_2e-2_T_01e-4.json',
    'simulation_parameters': (
        'n040e3_conf_020e0_001e3_iso_002e3_eq_015e3_ens_005e3.json'
    ),
}
SIMULATION_2 = {
    'immutables': IMMUTABLES,
    'initials': 'from_file_13e-1.json',
    'externals': 'velocity_scaling_HV_4e-4_T_01e-4.json',
    'simulation_parameters': (
        'n002e6_conf_001e3_500e3_iso_100e3_eq_015e3_ens_005e3.json'
    ),
}

CONFIGS = [
    [SIMULATION_1, ],
    [SIMULATION_2, ],
]

for i, configs in enumerate(CONFIGS):
    PATH_TO_DATA = os.path.join(os.getcwd(), 'data', get_date())
    print(PATH_TO_DATA)
    main(
        config_filenames=configs,
        is_with_isotherms=True,
    )
    SUFFIX = '+'.join(
        [item.get('externals').removesuffix('.json') for item in configs]
    )
    try:
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{SUFFIX}')
    except FileNotFoundError:
        PATH_TO_DATA = os.path.join(os.getcwd(), 'data', get_date())
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{SUFFIX}')
