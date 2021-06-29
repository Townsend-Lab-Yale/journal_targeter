import os
import shutil
import logging
from pathlib import Path
from importlib import resources

import dotenv
from flask import Flask

from . import paths


_logger = logging.getLogger(__name__)


def load_dotenv_as_dict():
    """Load env file values as ordered dictionary."""
    env_path = paths.ENV_PATH
    # if not os.path.exists(env_path):
    #     defaults = {''}
    #     for key in defaults:
    #         dotenv.set_key(env_path, key, defaults[key])
    dotenv_vals = dotenv.dotenv_values(env_path)
    # dotenv.load_dotenv(env_path, verbose=False)
    # if 'FLASK_APP' not in os.envir0n:
    #     os.envir0n['FLASK_APP'] = 'journal_choice.journals'
    return dotenv_vals


def refresh_data(app: Flask = None, rebuild_scopus=False):
    from .reference import TM
    from .demo import update_demo_plot
    TM.refresh_matching_data(rebuild_scopus=rebuild_scopus)
    with app.app_context():
        demo_prefix = app.config['DEMO_PREFIX']
    update_demo_plot(demo_prefix, use_pickle=False)


def copy_initial_data():
    os.makedirs(paths.DATA_ROOT, exist_ok=True)
    os.makedirs(paths.METRICS_DIR, exist_ok=True)
    os.makedirs(paths.PUBMED_DIR, exist_ok=True)
    added_data = []
    for dir_name, file_name in [
        ('metrics', 'jcr_meta.tsv.gz'),
        ('metrics', 'scopus_meta.tsv.gz'),
        ('metrics', 'scopus_map.tsv.gz'),
        ('pubmed', 'meta.tsv.gz'),
        ('demo', 'sars.ris'),
        ('demo', 'sars.yaml'),
        ('demo', 'example.yaml'),
    ]:
        new_path = Path(paths.DATA_ROOT).joinpath(dir_name, file_name)
        if not new_path.exists():
            added_data.append(file_name)
            with resources.path(f'journal_choice.refs.{dir_name}', file_name) as path:
                shutil.copy(path, new_path)
    if added_data:
        _logger.info(f"Copied reference data to {paths.DATA_ROOT}: {added_data}.")
