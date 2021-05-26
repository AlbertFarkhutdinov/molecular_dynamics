from datetime import date
import os
from scripts.main import main

filenames = [
    'cooling_T_1.0_HV_8e-2.json',
    'cooling_T_1.0_HV_6e-2.json',
    'cooling_T_1.0_HV_4e-2.json',
    'cooling_T_1.0_HV_1e-2.json',
    'cooling_T_1.0_HV_3e-2.json',
    'cooling_T_1.0_HV_5e-2.json',
    'cooling_T_1.0_HV_7e-2.json',
    'cooling_T_1.0_HV_9e-2.json',
]
for filename in filenames:
    try:
        PATH_TO_DATA = os.path.join(os.getcwd(), 'data', str(date.today()))
        print(PATH_TO_DATA)
        main(
            config_filenames=[filename],
            is_with_isotherms=True,
        )
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{filenames[0][:-5]}')
    except FileNotFoundError:
        PATH_TO_DATA = os.path.join(os.getcwd(), 'data', str(date.today()))
        print(PATH_TO_DATA)
        main(
            config_filenames=[filename],
            is_with_isotherms=True,
        )
        os.rename(PATH_TO_DATA, f'{PATH_TO_DATA}_{filenames[0][:-5]}')
