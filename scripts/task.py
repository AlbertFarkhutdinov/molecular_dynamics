from helpers import get_date
import os
from main import main
from scripts.log_config import logger

IMMUTABLES = 'lennard_jones.json'
SIMULATION_1 = {
    'immutables': IMMUTABLES,
    'initials': 'from_file_13e-1.json',
    'externals': 'velocity_scaling_HV_2e-2_T_13e-1.json',
    'simulation_parameters': (
        'n010e0_conf_005e0_100e0_iso_500e0_eq_500e0_ens_002e3.json'
    ),
}

CONFIGS = [
    [SIMULATION_1, ],
    # [SIMULATION_1, ],
]

for i, configs in enumerate(CONFIGS):
    PATH_TO_DATA = os.path.join(
        os.path.dirname(os.getcwd()),
        'data',
        get_date(),
    )
    print(PATH_TO_DATA)
    main(
        config_filenames=configs,
        is_with_isotherms=True,
    )
    logger.remove()
    suffix = '+'.join(
        [item.get('externals').removesuffix('.json') for item in configs]
    )
    try:
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{suffix}')
    except FileNotFoundError:
        PATH_TO_DATA = os.path.join(
            os.path.dirname(os.getcwd()),
            'data',
            get_date(),
        )
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{suffix}')
