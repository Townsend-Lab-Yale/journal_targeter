import os
import pathlib

import appdirs


_app_dirs = appdirs.AppDirs('journal_choice')
DATA_ROOT = _app_dirs.user_data_dir
CONFIG_DIR = pathlib.Path(_app_dirs.user_config_dir).joinpath('config')
ENV_PATH = CONFIG_DIR.joinpath('.env')
DEMO_DIR = pathlib.Path(_app_dirs.user_data_dir).joinpath('demo')
METRICS_DIR = os.path.join(DATA_ROOT, 'metrics')
PUBMED_DIR = os.path.join(DATA_ROOT, 'pubmed')
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PUBMED_DIR, exist_ok=True)
