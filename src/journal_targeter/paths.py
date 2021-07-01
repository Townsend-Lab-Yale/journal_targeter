import os
import pathlib

import appdirs


_app_dirs = appdirs.AppDirs('journal_targeter')
DATA_ROOT = _app_dirs.user_data_dir
CONFIG_DIR = pathlib.Path(_app_dirs.user_config_dir).joinpath('config')
ENV_PATH = CONFIG_DIR.joinpath('.env')
DEMO_DIR = pathlib.Path(_app_dirs.user_data_dir).joinpath('demo')
METRICS_DIR = os.path.join(DATA_ROOT, 'metrics')
PUBMED_DIR = os.path.join(DATA_ROOT, 'pubmed')
SESSION_DIR = os.path.join(DATA_ROOT, 'sessions')
# pubmed files
PM_MEDLINE_PATH = os.path.join(DATA_ROOT, 'pubmed', 'J_Medline.txt')
PM_META_PATH = os.path.join(DATA_ROOT, 'pubmed', 'meta.tsv.gz')
TM_PICKLE_PATH = os.path.join(DATA_ROOT, 'pubmed', 'tm.pickle.gz')

os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PUBMED_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)
