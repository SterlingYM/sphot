import tomllib
import os
from pathlib import Path
from .logging import logger

default_config_path = Path(__file__).parent / 'default_config.toml'

def load_config(user_config_path = 'sphot_config.toml'):
    # read default config
    with open(default_config_path, 'rb') as f:
        config = tomllib.load(f)

    if os.path.exists(user_config_path):
        with open(user_config_path, 'rb') as f:
            user_config = tomllib.load(f)
            config.update(user_config)
        logger.info(f'User config file loaded: {user_config_path}')
    else:
        logger.info(f'Using the default config file: {default_config_path}')
    return config

config = load_config(user_config_path = 'sphot_config.toml')