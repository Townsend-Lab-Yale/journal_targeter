import os
import shutil
import logging
from pathlib import Path
from importlib import resources

import nltk
from flask import Flask

from . import paths
from .app import db, source_tracker
from .app.models import Source, REFS_PKG


_logger = logging.getLogger(__name__)
_REF_FILES = [
        ('metrics', 'jcr_meta.tsv.gz', 'jcr'),
        ('metrics', 'scopus_meta.tsv.gz', 'scopus'),
        ('metrics', 'scopus_map.tsv.gz', 'scopus'),
        ('pubmed', 'meta.tsv.gz', 'pubmed'),
        ('demo', 'sars.ris', 'app'),
        ('demo', 'sars.yaml', 'app'),
        ('demo', 'example.yaml', 'app'),
        ('romeo', 'sherpa_romeo_map.tsv.gz', 'romeo'),
        ('doaj', 'doaj.tsv.gz', 'doaj'),
    ]


def refresh_data(app: Flask = None, rebuild_scopus=False):
    from .reference import TM
    from .demo import update_demo_plot
    TM.refresh_matching_data(rebuild_scopus=rebuild_scopus)
    with app.app_context():
        demo_prefix = app.config['DEMO_PREFIX']
    update_demo_plot(demo_prefix, use_jane_tables=False)


def copy_initial_data(app):
    os.makedirs(paths.DATA_ROOT, exist_ok=True)
    os.makedirs(paths.METRICS_DIR, exist_ok=True)
    os.makedirs(paths.PUBMED_DIR, exist_ok=True)
    added_data = []
    for dir_name, file_name, source in _REF_FILES:
        new_path = Path(paths.DATA_ROOT).joinpath(dir_name, file_name)
        if source_tracker.is_repo_newer_or_not_in_local_db(source) \
                or not new_path.exists():
            resource_dir = f'{REFS_PKG}.{dir_name}' if dir_name else REFS_PKG
            with resources.path(resource_dir, file_name) as path:
                shutil.copy2(path, new_path)
            added_data.append(file_name)
            # SET NEW DATE IN DB
            with app.app_context():
                Source.updated_at(source_name=source,
                                  update_time=source_tracker.dates_repo[source])
    if added_data:
        source_tracker.refresh_dates_user(app)
        _logger.info(f"Copied reference data to {paths.DATA_ROOT}: {added_data}.")
    nltk.download('wordnet', download_dir=paths.NLTK_DIR, quiet=True)


def retire_and_backup_source(source_name, clear_map=True):
    """Move source data to new file with .prev suffix, optionally keeping ID map."""
    clear_paths = []
    keep_paths = []
    if source_name == 'pubmed':
        clear_paths.extend([paths.PM_META_PATH, paths.TM_PICKLE_PATH])
    elif source_name == 'doaj':
        clear_paths.append(paths.DOAJ_PATH)
    elif source_name == 'romeo':
        clear_paths.append(paths.ROMEO_MAP_PATH)
    else:  # handle metric sources
        map_path = os.path.join(paths.METRICS_DIR, f"{source_name}_map.tsv.gz")
        meta_path = os.path.join(paths.METRICS_DIR, f"{source_name}_meta.tsv.gz")
        clear_paths.append(meta_path)
        if clear_map:
            clear_paths.append(map_path)
        else:
            keep_paths.append(map_path)
    for path in clear_paths:
        if os.path.exists(path):
            shutil.move(path, path + '.prev')
    for path in keep_paths:
        if os.path.exists(path):
            shutil.copy2(path, path + '.prev')
