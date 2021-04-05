"""
This module is used for checking
the project code for compliance with PEP8.

"""


import pylint.lint
pylint_opts = [
    '--ignore-imports=yes',
    '--disable=C0114, C0115, C0116, W0511',
    '__init__.py',
    'pylint_inspection.py',
    'scripts_new\\integrators\\__init__.py',
    'scripts_new\\integrators\\base_integrator.py',
    'scripts_new\\integrators\\npt_mttk.py',
    'scripts_new\\integrators\\npt_nose_hoover.py',
    'scripts_new\\integrators\\nve.py',
    'scripts_new\\integrators\\nvt_velocity_scaling.py',
    'scripts_new\\potentials\\__init__.py',
    'scripts_new\\potentials\\base_potential.py',
    'scripts_new\\potentials\\dzugutov.py',
    'scripts_new\\potentials\\lennard_jones.py',
    'scripts_new\\properties\\__init__.py',
    'scripts_new\\properties\\radial_distribution_function.py',
    'scripts_new\\properties\\static_structure_factor.py',
    'scripts_new\\properties\\transport_properties.py',
    'scripts_new\\__init__.py',
    'scripts_new\\accelerations_calculator.py',
    'scripts_new\\constants.py',
    'scripts_new\\core.py',
    'scripts_new\\dynamic_parameters.py',
    'scripts_new\\external_parameters.py',
    'scripts_new\\helpers.py',
    'scripts_new\\immutable_parameters.py',
    'scripts_new\\initializer.py',
    'scripts_new\\isotherm.py',
    'scripts_new\\log_config.py',
    'scripts_new\\main.py',
    # 'scripts_new\\numba_procedures.py',
    'scripts_new\\plotter.py',
    'scripts_new\\saver.py',
    'scripts_new\\system.py',
]
pylint.lint.Run(pylint_opts)
pylint_opts = [
]
pylint.lint.Run(pylint_opts)
