import logging

from .models import MasterTable
from .pubmed import load_pubmed_journals, TitleMatcher

MT = MasterTable()  # key metadata for nlmids. populate with MT.init_data(pm_ext)
TM = TitleMatcher()  # matches journal data to nlmids.

_logger = logging.getLogger(__name__)


def init_reference_data_from_cache():
    _logger.info("Loading reference data for matching and metadata lookups.")
    pm_ext = load_pubmed_journals(refresh=False)
    MT.init_data(pm_ext)
    TM.init_data()  # use existing data. priority: pickle > tsv.gz