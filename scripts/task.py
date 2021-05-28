from datetime import date
import os
from scripts.main import main
from scripts.log_config import logger

filenames = [
    'cooling_T_0.3_HV_8e-2.json',
    'cooling_T_0.3_HV_9e-2.json',
]

for i, filename in enumerate(filenames):
    PATH_TO_DATA = os.path.join(
        os.path.dirname(os.getcwd()),
        'data',
        str(date.today()),
    )
    print(PATH_TO_DATA)
    main(
        config_filenames=[filename],
        is_with_isotherms=True,
    )
    logger.remove()
    try:
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{filename[:-5]}')
    except FileNotFoundError:
        PATH_TO_DATA = os.path.join(
            os.path.dirname(os.getcwd()),
            'data',
            str(date.today()),
        )
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{filename[:-5]}')
