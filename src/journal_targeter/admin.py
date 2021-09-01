import os
import shutil
import logging
import datetime
from pathlib import Path
from importlib import resources
from collections import defaultdict

import nltk
import yaml
from flask import Flask

from . import paths
from .app import db
from .app.models import Source


_logger = logging.getLogger(__name__)
_REF_FILES = [
        ('metrics', 'jcr_meta.tsv.gz', 'jcr'),
        ('metrics', 'scopus_meta.tsv.gz', 'scopus'),
        ('metrics', 'scopus_map.tsv.gz', 'scopus'),
        ('pubmed', 'meta.tsv.gz', 'pubmed'),
        ('demo', 'sars.ris', 'na'),
        ('demo', 'sars.yaml', 'na'),
        ('demo', 'example.yaml', 'na'),
        ('romeo', 'sherpa_romeo_map.tsv.gz', 'romeo'),
        ('doaj', 'doaj.tsv.gz', 'doaj'),
        ('', 'updates.yaml', 'na'),
    ]
REFS_PKG = 'journal_targeter.refs'


def refresh_data(app: Flask = None, rebuild_scopus=False):
    from .reference import TM
    from .demo import update_demo_plot
    TM.refresh_matching_data(rebuild_scopus=rebuild_scopus)
    with app.app_context():
        demo_prefix = app.config['DEMO_PREFIX']
    update_demo_plot(demo_prefix, use_pickle=False)


def copy_initial_data(app):
    os.makedirs(paths.DATA_ROOT, exist_ok=True)
    os.makedirs(paths.METRICS_DIR, exist_ok=True)
    os.makedirs(paths.PUBMED_DIR, exist_ok=True)
    with app.app_context():
        db.create_all()
    dates_repo = _get_source_dates_repo()
    dates_user = _get_source_dates_user(app)
    repo_newer = defaultdict(lambda: True)
    for source in dates_repo:
        if source in dates_user and dates_user[source] >= dates_repo[source]:
            repo_newer[source] = False
    repo_newer['na'] = False

    added_data = []
    for dir_name, file_name, source in _REF_FILES:
        new_path = Path(paths.DATA_ROOT).joinpath(dir_name, file_name)
        if repo_newer[source] or not new_path.exists():
            resource_dir = f'{REFS_PKG}.{dir_name}' if dir_name else REFS_PKG
            with resources.path(resource_dir, file_name) as path:
                shutil.copy2(path, new_path)
            added_data.append(file_name)
            # SET NEW DATE IN DB (if in valid source)
            if source not in dates_repo:
                continue
            with app.app_context():
                Source.store_update(source_name=source,
                                    update_time=dates_repo[source])
    if added_data:
        _logger.info(f"Copied reference data to {paths.DATA_ROOT}: {added_data}.")
    nltk.download('wordnet', download_dir=paths.NLTK_DIR, quiet=True)


def _get_source_dates_repo():
    with resources.path(REFS_PKG, 'updates.yaml') as path:
        with open(path, 'r') as infile:
            dates_repo = yaml.load(infile, yaml.SafeLoader)

    for source, date_str in dates_repo.items():
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        dates_repo[source] = dt
    return dates_repo


def _get_source_dates_user(app):
    with app.app_context():
        recs = Source.query.all()
    dates_user = dict()
    for r in recs:
        dates_user[r.source_name] = r.update_time
    return dates_user
