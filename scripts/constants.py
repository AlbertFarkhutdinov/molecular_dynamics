import os


BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_CONFIG = os.path.join(BASE_DIRECTORY, 'config_files')
PATH_TO_DATA = os.path.join(BASE_DIRECTORY, 'data')
PATH_TO_PLOTS = os.path.join(BASE_DIRECTORY, 'plots')
TEMPERATURE_MINIMUM = 1e-5
IS_LOGGED = True


if __name__ == '__main__':
    print(BASE_DIRECTORY)
    print(PATH_TO_CONFIG)
    print(PATH_TO_DATA)
