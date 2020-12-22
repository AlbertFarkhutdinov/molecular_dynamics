"""
Liquid argon:
_____________
mass = 6.69E-26 kilograms
distance = sigma = 0.341E-9 meters
energy = epsilon = 1.65E-21 joules
temperature = epsilon / k_B = 119.8 kelvin
tau = sigma * sqrt(mass / epsilon) = 2.17E-12 seconds
velocity = sigma / tau = 1.57E2 m/s
force = epsilon/sigma = 4.85E-12 newtons
pressure = epsilon / (sigma ^ 3) = 4.2E7 pascal = 4.2E2 atmospheres

"""


from numpy import set_printoptions

from scripts.core import MolecularDynamics


def main(
        config_filename: str = None,
        is_initially_frozen: bool = True,
        is_rdf_calculated: bool = True,
):
    MolecularDynamics(
        config_filename=config_filename,
        is_initially_frozen=is_initially_frozen,
        is_rdf_calculated=is_rdf_calculated
    ).run_md()


if __name__ == '__main__':
    # TODO check potential at T = 2.8 (compare 2020-12-17 and the book, p.87)
    set_printoptions(threshold=5000)

    CONFIG_FILE_NAME = 'book_chapter_4_stage_1.json'
    # CONFIG_FILE_NAME = 'book_chapter_4_stage_2.json'
    # CONFIG_FILE_NAME = 'cooling.json'
    # CONFIG_FILE_NAME = 'equilibrium_2.8.json'
    # CONFIG_FILE_NAME = 'equilibrium_0.01.json'

    main(
        config_filename=CONFIG_FILE_NAME,
        is_initially_frozen=False,
        is_rdf_calculated=True,
    )
