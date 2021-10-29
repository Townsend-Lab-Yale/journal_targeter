import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .paths import DOAJ_PATH

_logger = logging.getLogger(__name__)


def match_and_trim_doaj_csv(path_csv: str, n_processes: Optional[int]) -> pd.DataFrame:
    from .reference import TM
    if not n_processes:
        n_processes = os.environ.get('N_PROCESSES', 1)
    df = pd.read_csv(path_csv, low_memory=False)
    _logger.info("Loaded DOAJ data. Start title/ISSN matching.")
    match = TM.lookup_uids_from_title_issn(df['Journal title'],
                                           df['Journal ISSN (print version)'],
                                           df['Journal EISSN (online version)'],
                                           n_processes=n_processes)
    df['nlmid'] = match['uid'].values
    keep_dict = dict([
        ('Journal title', 'title'),  # npj Biofilms and Microbiomes
        ('URL in DOAJ', 'url_doaj'),  # https://doaj.org/toc/c2f560adb7ae427ea563fa6508f44aa8
        ('Journal license', 'license'),  # CC BY
        ('Author holds copyright without restrictions', 'author_copyright'),  # Yes  # author retains unrestricted copyrights and publishing rights
        ('Average number of weeks between article submission and publication', 'n_weeks_avg'),  # 9
        ('APC', 'apc'),  # Yes
        ('APC amount', 'apc_val'),  # 3490 USD  # highest fee charged by journal
        ('Preservation Services', 'preservation'),  # CLOCKSS, LOCKSS, PMC
        ("Does the journal comply to DOAJ's definition of open access?", 'doaj_compliant'),  # Yes
        ('DOAJ Seal', 'doaj_seal'),  # Yes
    ])

    doaj = df.loc[df.nlmid.notnull(), ['nlmid'] + list(keep_dict.keys())].rename(columns=keep_dict)
    doaj.set_index('nlmid', inplace=True)
    doaj.to_csv(DOAJ_PATH, sep='\t', index=True, compression='gzip',
                encoding='utf8', line_terminator='\n')
    return doaj


def load_doaj_table():
    d = pd.read_csv(DOAJ_PATH, sep='\t', compression='gzip',
                    lineterminator='\n', encoding='utf8')
    d = d.set_index('nlmid').drop(columns=['title'])
    d['doaj_score'] = _get_doaj_score(d)
    return d


def _get_doaj_score(df) -> pd.Series:
    """Get score for DOAJ metrics (high = good) within df table."""
    temp = df.copy()
    temp['has_preservation'] = temp['preservation'].notnull()
    temp_sorted = temp.sort_values(['doaj_seal', 'author_copyright', 'has_preservation', 'n_weeks_avg'], ascending=[False, False, False, True]).copy()
    temp_sorted['doaj_score'] = np.arange(len(temp), 0, -1)
    return temp_sorted['doaj_score']
