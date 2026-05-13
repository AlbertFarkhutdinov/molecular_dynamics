from pathlib import Path


BASE_DIRECTORY = Path(__file__).parents[2]
PATH_TO_CONFIG = BASE_DIRECTORY.joinpath('config_files')
PATH_TO_DATA = BASE_DIRECTORY.joinpath('data')
PATH_TO_PLOTS = BASE_DIRECTORY.joinpath('plots')
TEMPERATURE_MINIMUM = 1e-5
IS_LOGGED = True


if __name__ == '__main__':
    print(BASE_DIRECTORY)
    print(PATH_TO_CONFIG)
    print(PATH_TO_DATA)
