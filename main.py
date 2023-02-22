import numpy as np

from md_pipelines import MolecularDynamicsSteps
from common.helpers import get_json, save_config_parameters


def main(
        config_filenames: list[dict[str, str]],
        is_with_isotherms: bool = True,
):
    _config_filename = config_filenames[0]
    md_instance = MolecularDynamicsSteps(
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
    SIMULATION_0 = {
        'immutables': IMMUTABLES,
        'initials': 'from_crystal_01e-1.json',
        'externals': 'velocity_scaling_HV_2e-5_T_01e-4.json',
        'simulation_parameters': (
            'n010e0_conf_005e0_100e0_iso_500e0_eq_015e3_ens_030e3.json'
        ),
    }
    # SIMULATION_1 = {
    #     'immutables': IMMUTABLES,
    #     'initials': 'from_file_03e-1.json',
    #     'externals': 'velocity_scaling_HV_2e-5_T_01e-4.json',
    #     'simulation_parameters': (
    #         'n002e6_conf_001e3_500e3_iso_100e3_eq_015e3_ens_005e3.json'
    #     ),
    # }
    main(
        config_filenames=[
            SIMULATION_0,
        ],
        is_with_isotherms=True,
    )
