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
    'scripts\\integrators\\__init__.py',
    'scripts\\integrators\\base_integrator.py',
    'scripts\\integrators\\npt_mttk.py',
    'scripts\\integrators\\npt_nose_hoover.py',
    'scripts\\integrators\\nve.py',
    'scripts\\integrators\\nvt_velocity_scaling.py',
    'scripts\\potentials\\__init__.py',
    'scripts\\potentials\\base_potential.py',
    'scripts\\potentials\\dzugutov.py',
    'scripts\\potentials\\lennard_jones.py',
    'scripts\\properties\\__init__.py',
    'scripts\\properties\\radial_distribution_function.py',
    'scripts\\properties\\static_structure_factor.py',
    'scripts\\properties\\transport_properties.py',
    'scripts\\__init__.py',
    'scripts\\accelerations_calculator.py',
    'scripts\\constants.py',
    'scripts\\core.py',
    'scripts\\dynamic_parameters.py',
    'scripts\\external_parameters.py',
    'scripts\\helpers.py',
    'scripts\\immutable_parameters.py',
    'scripts\\initializer.py',
    'scripts\\isotherm.py',
    'scripts\\log_config.py',
    # 'scripts\\numba_procedures.py',
    'scripts\\plotter.py',
    'scripts\\saver.py',
    'scripts\\system.py',
    'main.py',
]
pylint.lint.Run(pylint_opts)
pylint_opts = [
]
pylint.lint.Run(pylint_opts)
