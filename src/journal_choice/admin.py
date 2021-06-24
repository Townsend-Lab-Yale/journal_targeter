import os
import shutil
import logging
from pathlib import Path
from importlib import resources

from .paths import DATA_ROOT, METRICS_DIR, PUBMED_DIR


_logger = logging.getLogger(__name__)


def refresh_data(rebuild_scopus=False):
    from .reference import TM
    from .demo import update_demo_plot
    TM.refresh_matching_data(rebuild_scopus=rebuild_scopus)
    demo_prefix = os.environ.get('DEMO_PREFIX', 'default')
    update_demo_plot(demo_prefix, use_pickle=False)


def copy_initial_data():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PUBMED_DIR, exist_ok=True)
    added_data = []
    for dir_name, file_name in [
        ('metrics', 'jcr_meta.tsv.gz'),
        ('metrics', 'scopus_meta.tsv.gz'),
        ('metrics', 'scopus_map.tsv.gz'),
        ('pubmed', 'meta.tsv.gz'),
        ('demo', 'sars.ris'),
        ('demo', 'sars.yaml'),
    ]:
        new_path = Path(DATA_ROOT).joinpath(dir_name, file_name)
        if not new_path.exists():
            added_data.append(file_name)
            with resources.path(f'journal_choice.refs.{dir_name}', file_name) as path:
                shutil.copy(path, new_path)
    if added_data:
        _logger.info(f"Copied reference data to {DATA_ROOT}: {added_data}.")
    else:
        _logger.info(f"Config files found in {DATA_ROOT}.")
