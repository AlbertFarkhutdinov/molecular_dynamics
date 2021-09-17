from helpers import get_date
import os
from main import main
from scripts.log_config import logger

configs = [
    # ['cooling_normal_1.3_normal.json', ],
    ['cooling_normal_1.3_long.json', ],
]

for i, config in enumerate(configs):
    PATH_TO_DATA = os.path.join(
        os.path.dirname(os.getcwd()),
        'data',
        str(get_date),
    )
    print(PATH_TO_DATA)
    main(
        config_filenames=config,
        is_with_isotherms=True,
    )
    logger.remove()
    suffix = '+'.join([item.removesuffix('.json') for item in config])
    try:
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{suffix}')
    except FileNotFoundError:
        PATH_TO_DATA = os.path.join(
            os.path.dirname(os.getcwd()),
            'data',
            str(get_date),
        )
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{suffix}')
