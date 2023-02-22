"""The module for PEP8 check."""


from pylint_af import PyLinter

from common.constants import BASE_DIR


if __name__ == '__main__':
    PyLinter(
        root_directory=str(BASE_DIR),
        ignored_statements={
            'C0114',
            'C0115',
            'C0116',
            'E1133',
            'W0511',
        },
        ignored_paths={
            'common',
            # 'configurations',
            'core',
            'drawing',
            # 'initializers',
            # 'inspections',
            'integrators',
            # 'logs',
            'md_pipelines',
            # 'optimizers',
            # 'potentials',
            'properties',
        }
    ).check()
