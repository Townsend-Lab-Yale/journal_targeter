from .models import MasterTable
from .pubmed import load_pubmed_journals, TitleMatcher

pm_ext = load_pubmed_journals(refresh=False)
MT = MasterTable(pm_ext)
TM = TitleMatcher()
