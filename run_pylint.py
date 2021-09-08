"""
This module is used for checking the project code for compliance with PEP8.

"""

from pylint_af import PyLinter


if __name__ == '__main__':
    PyLinter(
        is_printed=False,
        # ignored_paths={'scripts\\numba_procedures.py'},
        ignored_statements={
            'C0114',
            'C0115',
            'C0116',
            'W0511',
        },
    ).check()
