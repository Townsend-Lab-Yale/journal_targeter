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
    with app.app_context():
        db.create_all()
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


def backup_and_clear_pm_metadata():
    for path in [paths.PM_META_PATH, paths.TM_PICKLE_PATH]:
        if os.path.exists(path):
            backup_path = path + '.prev'
            shutil.move(path, backup_path)
