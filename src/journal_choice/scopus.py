""""Scopus Sources table via https://www.scopus.com/sources"""
import os
import numpy as np
import pandas as pd

from . import DATA_DIR
from .helpers import get_issn_safe, get_issn1, get_issn_comb, get_clean_lowercase


SCOPUS_PATH = os.path.join(DATA_DIR, 'scopus', 'ext_list_october_2019.xlsx')
CITESCORE_YEAR = '2018'  # @TODO: Identify latest year in sources xlsx file

_COL_RENAME_DICT = {
    'Sourcerecord id': 'scopus_id',
    'Source Title (Medline-sourced journals are indicated in Green)': 'journal_name',
    'Article language in source (three-letter ISO language codes)': 'lang',
    f'{CITESCORE_YEAR} CiteScore': 'citescore_latest',
    'Medline-sourced Title? (see more info under separate tab)': 'is_medline',
    'Publisher imprints grouped to main Publisher': 'imprints',
    "Publisher's Country/Territory": 'publisher_region',
    "Open Access status, i.e., registered in DOAJ and/or ROAD. Status September 2019":
        'open_access_status',

}


def load_scopus_journals_full(scopus_xlsx_path=SCOPUS_PATH):
    """Load Scopus journals table.

    Args:
        scopus_xlsx_path (str): path to Scopus journals file.

    Returns:
        pd.DataFrame
    """
    df = pd.read_excel(scopus_xlsx_path,
                       dtype={'Print-ISSN': str, 'E-ISSN': str})
    # sheet_name='Scopus Sources September 2019'
    df.columns = [i.replace('\n', ' ').replace('  ', ' ').strip() for i in df.columns]
    df.rename(columns=_COL_RENAME_DICT, inplace=True)
    return df


def load_scopus_journals_reduced(scopus_xlsx_path=SCOPUS_PATH):
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
                 #  'Active or Inactive',
                 #  'Titles discontinued by Scopus due to quality issues',
                 #  'Coverage',
                 'lang',
                 #  '2016 CiteScore',
                 #  '2017 CiteScore',
                 'citescore_latest',
                 'is_medline',
                 'open_access_status',
                 #  'Articles in Press included?',
                 #  'Added to list September 2019',
                 'Source Type',
                 #  'Title history indication',
                 #  'Related title to title history indication',
                 #  'Other related title 1',
                 #  'Other related title 2',
                 #  'Other related title 3',
                 "Publisher's Name",
                 'imprints',
                 'publisher_region',
                 ]
    df = load_scopus_journals_full(scopus_xlsx_path)
    dfs = df.loc[df['Active or Inactive'] == 'Active', keep_cols].copy()
    dfs['journal_name'] = dfs['journal_name'].str.strip()
    dup_titles = (dfs['journal_name'].value_counts() > 1) \
        .loc[lambda v: v].index
    dfs['is_unique_title'] = ~dfs['journal_name'].isin(dup_titles)
    dfs['Print-ISSN'] = dfs['Print-ISSN'].replace(np.nan, '').str.strip()
    dfs['E-ISSN'] = dfs['E-ISSN'].replace(np.nan, '').str.strip()
    dfs['issn_print'] = get_issn_safe(dfs['Print-ISSN'])
    dfs['issn_online'] = get_issn_safe(dfs['E-ISSN'])
    dfs['issn1'] = get_issn1(dfs['issn_print'], dfs['issn_online'])
    dfs['issn_comb'] = get_issn_comb(dfs['issn_print'], dfs['issn_online'])
    dfs['title_safe'] = dfs['journal_name'].apply(get_clean_lowercase)
    dup_titles_safe = (dfs['title_safe'].value_counts() > 1) \
        .loc[lambda v: v].index
    dfs['is_unique_title_safe'] = ~dfs['title_safe'].isin(dup_titles_safe)
    dfs.set_index('scopus_id', inplace=True)
    return dfs
