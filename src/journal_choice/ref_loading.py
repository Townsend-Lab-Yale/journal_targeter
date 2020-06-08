import logging

import numpy as np
import pandas as pd
from RISparser import readris
from RISparser.config import TAG_KEY_MAPPING


_logger = logging.getLogger(__name__)
_logger.setLevel('DEBUG')


def _first_valid_title(title_series):
    titles = title_series.dropna()
    n_options = titles.size
    title = titles.iloc[0] if n_options else np.nan
    return title


# journal name in T2 (paperpile), JF (mendeley)
_NAME_FIELD_PREFERENCE = [
    'J1',  # periodical name user abbrev 1
    'J2',  # alternate / abbreviated title
    'JA',  # name standard abbrev
    'JF',  # name full format
    'JO',  # name full format
    'T2',  # secondary / journal title
]

_TYPE_COL = TAG_KEY_MAPPING['TY']
_DUP_FIELDS = ['DO', 'TI']
_USE_TYPES = {'JOUR', 'JFULL'}


def read_ris_file(ris_path):
    """Load table of references from RIS file path."""
    with open(ris_path, 'r') as bibfile:
        entries = readris(bibfile)
    df = pd.DataFrame.from_records(entries)
    _logger.debug(f"Loaded {len(df)} entries via RIS.")
    # RESTRICT TO JOURNAL ROWS
    skipped_types = set(df[_TYPE_COL]).difference(_USE_TYPES)
    if skipped_types:
        _logger.debug(f"Skipped {len(skipped_types)} entry type(s): {skipped_types}")
    df = df[df[_TYPE_COL].isin(_USE_TYPES)]
    # IDENTIFY JOURNAL NAMES
    name_col_preference = [TAG_KEY_MAPPING[i] for i in _NAME_FIELD_PREFERENCE
                           if i in TAG_KEY_MAPPING]
    j_name_options = [i for i in df.columns if i in name_col_preference]
    _logger.debug(f"Title column options: {j_name_options}")
    df.insert(0, 'journal',
              df[j_name_options].apply(_first_valid_title, axis=1))
    # DEDUPLICATE, ignoring all-nan matching fields
    dup_cols = [TAG_KEY_MAPPING[i] for i in _DUP_FIELDS
                if i in TAG_KEY_MAPPING]
    check_inds = (~df[dup_cols].isnull().all(axis=1)).loc[lambda v: v].index
    checkable = len(check_inds) > 1
    if checkable:
        keep_locs = df.loc[check_inds].drop_duplicates(subset=dup_cols, keep='first').index
        drop_inds = set(check_inds).difference(keep_locs)
        n_dups = len(drop_inds)
        _logger.debug(f"Dropped {n_dups} duplicate rows.")
        df = df.drop(drop_inds)
    else:
        _logger.debug("No suitable columns to check for duplicates.")
    return df


def match_journals(tm, journal_counts):
    """Get UIDs (and matching methods) for parsed user input.

    Args:
        tm (TitleMatcher): pubmed TitleMatcher for matching.
        journal_counts (Union[pd.Series, dict, OrderedDict, Counter]):
            map from journal name (index/keys) to citation count (values)
    """
    if type(journal_counts) is pd.Series:
        user_titles = list(journal_counts.index)
        n_citations = list(journal_counts.values)
    else:
        user_titles = list(journal_counts.keys())
        n_citations = list(journal_counts.values())

    uids, methods = zip(*map(tm.get_uids_from_title, user_titles))
    m = pd.DataFrame({'user_title': user_titles, 'uids': uids,
                      'n_citations': n_citations,  'method': methods})
    unmatched_titles = list(m.loc[m.method == 'unmatched', 'user_title'])
    n_unmatched = len(unmatched_titles)
    n_matched = len(m) - n_unmatched
    _logger.debug(f"Successfully matched {n_matched} journals.")
    _logger.debug(f"{n_unmatched=}")
    _logger.debug(f"{unmatched_titles=}")
    return m
