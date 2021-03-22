"""
This module is used for checking
the project code for compliance with PEP8.

"""


import pylint.lint
pylint_opts = [
    '--ignore-imports=yes',
    '__init__.py',
    'pylint_inspection.py',
    'scripts\\__init__.py',
    'scripts\\constants.py',
    'scripts\\core.py',
    'scripts\\dynamic_parameters.py',
    'scripts\\external_parameters.py',
    'scripts\\helpers.py',
    'scripts\\isotherm.py',
    'scripts\\log_config.py',
    'scripts\\main.py',
    'scripts\\modeling_parameters.py',
    'scripts\\mtk_npt_integrator.py',
    'scripts\\nose_hoover.py',
    'scripts\\numba_procedures.py',
    'scripts\\plotter.py',
    'scripts\\potential_parameters.py',
    'scripts\\radial_distribution_function.py',
    'scripts\\saver.py',
    'scripts\\static_parameters.py',
    'scripts\\static_structure_factor.py',
    'scripts\\transport_properties.py',
    'scripts\\velocity_scaling.py',
    'scripts\\verlet.py',
]
pylint.lint.Run(pylint_opts)
pylint_opts = [
]
pylint.lint.Run(pylint_opts)
