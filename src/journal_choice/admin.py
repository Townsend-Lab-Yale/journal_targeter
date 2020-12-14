import os

from .pubmed import TM
from .demo import update_demo_plot


def refresh_data(rebuild_scopus=False):
    TM.refresh_matching_data(rebuild_scopus=rebuild_scopus)
    demo_prefix = os.environ.get('DEMO_PREFIX', 'default')
    update_demo_plot(demo_prefix, use_pickle=False)
