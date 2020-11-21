import os


BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_CONFIG = os.path.join(BASE_DIRECTORY, 'config_files')
PATH_TO_DATA = os.path.join(BASE_DIRECTORY, 'data')
TEMPERATURE_MINIMUM = 1e-15
IS_LOGGED = True
