import tomllib
import os
from pathlib import Path
from .logging import logger

user_config_path = 'sphot_conf.toml'
default_config_path = Path(__file__).parent / 'default_config.toml'

if os.path.exists(user_config_path):
    with open(user_config_path, 'rb') as f:
        config = tomllib.load(f)
    logger.info(f'User config file loaded: {user_config_path}')
else:
    with open(default_config_path, 'rb') as f:
        config = tomllib.load(f)
    logger.info(f'Using the default config file: {user_config_path}')