from pathlib import Path


BASE_DIR = Path('.').resolve().parent
CONFIG_DIR = BASE_DIR / 'config_files'
DATA_DIR = BASE_DIR / 'data'
PLOTS_DIR = BASE_DIR / 'plots'
TEMPERATURE_MINIMUM = 1e-5


if __name__ == '__main__':
    print(BASE_DIR)
    print(CONFIG_DIR)
    print(DATA_DIR)
    print(PLOTS_DIR)
