""""Load Scopus journals table.

Data via https://www.scopus.com/sources ('source titles only', sign in required).
"""
import os
import re
import time
import logging

import numpy as np
import pandas as pd

from .helpers import get_issn_safe, get_issn_comb, get_clean_lowercase


CITESCORE_YEAR = None  # Automatically identified from column names in sources file

_logger = logging.getLogger(__name__)


class ColumnException(Exception):
    pass


def load_scopus_journals_full(scopus_xlsx_path=None):
    """Load Scopus journals table.

    Args:
        scopus_xlsx_path (str): path to Scopus journals file.

    Returns:
        pd.DataFrame
    """
    global CITESCORE_YEAR
    _logger.info("Loading Scopus data from %s.", scopus_xlsx_path)
    time_start = time.perf_counter()
    df = pd.read_excel(scopus_xlsx_path,
                       dtype={'Print-ISSN': str, 'E-ISSN': str})
    time_end = time.perf_counter()
    _logger.info("Finished loading scopus data in %.1f seconds.", time_end - time_start)
    # sheet_name='Scopus Sources September 2019'
    df.columns = [i.replace('\n', ' ').replace('  ', ' ').strip() for i in df.columns]
    # COLUMN IDENTIFICATION
    id_col = _identify_column('sourcerecord', df.columns)
    name_col = _identify_column('source title', df.columns)
    oa_col = _identify_column('open access status', df.columns)
    lang_col = _identify_column('language', df.columns)
    imprints_col = _identify_column('imprints', df.columns)
    publisher_col = _identify_column('publisher', set(df.columns).difference({imprints_col}))
    type_col = _identify_column('type', df.columns)
    try:
        cs_cols = [(col, re.findall(r'\d{4}', col)[0]) for col in df.columns
                   if 'citescore' in col.lower()]
    except IndexError:
        raise IndexError("CiteScore columns don't match required pattern inc 4 digit year.")
    cs_cols.sort(key=lambda v: v[1], reverse=True)
    cs_col, CITESCORE_YEAR = cs_cols[0]
    df.rename(columns={cs_col: 'citescore',
                       name_col: 'journal_name',
                       oa_col: 'open_access_status',
                       lang_col: 'lang',
                       imprints_col: 'imprints',
                       publisher_col: 'publisher',
                       type_col: 'source_type',
                       }, inplace=True)
    df.insert(0, 'scopus_id', df[id_col].astype(str))
    return df


def load_scopus_journals_reduced(scopus_xlsx_path=None):
    """Load small / reduced Scopus journals table with key columns.

        Args:
            scopus_xlsx_path (str): path to Scopus journals file.

        Returns:
            pd.DataFrame
        """
    keep_cols = ['scopus_id',
                 'journal_name',
                 'Print-ISSN',
                 'E-ISSN',
                 'lang',
                 'citescore',
                 'open_access_status',
                 'source_type',
                 'publisher',
                 'imprints',
                 ]
    df = load_scopus_journals_full(scopus_xlsx_path)
    _logger.debug(f"Processing Scopus columns.")
    dfs = df.loc[df['Active or Inactive'] == 'Active', keep_cols].copy()
    dfs['journal_name'] = dfs['journal_name'].str.strip()
    dup_titles = (dfs['journal_name'].value_counts() > 1) \
        .loc[lambda v: v].index
    dfs['is_unique_title'] = ~dfs['journal_name'].isin(dup_titles)
    dfs['Print-ISSN'] = dfs['Print-ISSN'].replace(np.nan, '').str.strip()
    dfs['E-ISSN'] = dfs['E-ISSN'].replace(np.nan, '').str.strip()
    dfs['issn_print'] = get_issn_safe(dfs['Print-ISSN'])
    dfs['issn_online'] = get_issn_safe(dfs['E-ISSN'])
    dfs['issn_comb'] = get_issn_comb(dfs['issn_print'], dfs['issn_online'])
    dfs['title_safe'] = dfs['journal_name'].apply(get_clean_lowercase)
    dup_titles_safe = (dfs['title_safe'].value_counts() > 1) \
        .loc[lambda v: v].index
    dfs['is_unique_title_safe'] = ~dfs['title_safe'].isin(dup_titles_safe)
    dfs['is_open'] = dfs['open_access_status'].fillna('').str.lower().str.contains('doaj')
    dfs.set_index('scopus_id', inplace=True)
    return dfs


def _identify_column(substring_lower, columns):
    """Find column containing lowercase substring."""
    match_cols = [i for i in columns if substring_lower in i.lower()]
    if len(match_cols) != 1:
        raise ColumnException(f"No column with {substring_lower} in name.")
    return match_cols[0]
