import logging

import numpy as np
import pandas as pd
from RISparser import readris
from RISparser.config import TAG_KEY_MAPPING


_logger = logging.getLogger(__name__)
# _logger.setLevel('DEBUG')

# journal name in T2 (paperpile), JF (mendeley)
_NAME_FIELD_PREFERENCE = [
    'J1',  # periodical name user abbrev 1
    'J2',  # alternate / abbreviated title
    'JA',  # name standard abbrev
    'JF',  # name full format
    'JO',  # name full format
    'T2',  # secondary / journal title
]

_TITLE_TAG_OPTIONS = ['TI', 'T1']
_AUTHOR_TAG_OPTIONS = ['AU', 'A1']
_YEAR_TAG_OPTIONS = ['PY', 'Y1']

_TYPE_COL = TAG_KEY_MAPPING['TY']
_DUP_FIELDS = ['DO', 'TI', 'T1']  # columns used for identifying citation dups
_USE_TYPES = {'JOUR', 'JFULL'}  # ignore everything that isn't a journal article


class BadRisException(Exception):
    pass


def _get_journal_names(df):
    """Identify journal names from parsed RIS table."""
    options = _NAME_FIELD_PREFERENCE.copy()
    return _get_values_for_tag_options(df, tag_options=options)


def _get_article_titles(df):
    """Identify article titles from parsed RIS table."""
    options = _TITLE_TAG_OPTIONS.copy()
    return _get_values_for_tag_options(df, tag_options=options)


def _get_authors(df):
    """Identify author names from parsed RIS table."""
    options = _AUTHOR_TAG_OPTIONS.copy()
    authors = _get_values_for_tag_options(df, tag_options=options)
    # convert to list of strings, shortened e.g. by et al
    authors = [_shorten_authors(i) if type(i) is list else i for i in authors]
    return authors


def _shorten_authors(author_list):
    """Get short string for author list, e.g. Gaffney et al."""
    short_names = [i.split(',')[0] if ',' in i else i for i in author_list]
    if len(author_list) > 2:
        first_author = short_names[0]
        short = f'{first_author} et al'
        return short
    if len(author_list) == 2:
        ' & '.join(short_names)
    return author_list[0]


def _get_years(df):
    """Identify publication years from parsed RIS table."""
    options = _YEAR_TAG_OPTIONS.copy()
    years = _get_values_for_tag_options(df, tag_options=options)
    # convert to list to shorten cases like '2018///'
    years = [i[:4] if type(i) is str and len(i) > 4 else i for i in years]
    return years


def _get_values_for_tag_options(df, tag_options=None):
    """Extract first non-null value from columns corresponding to RIS tag options.

    Args:
        df (pd.DataFrame): parsed RIS table.
        tag_options (list): allowed tags corresponding to field, in order of
            preference. e.g. _NAME_FIELD_PREFERENCE for title
    """
    # IDENTIFY JOURNAL NAMES
    name_col_preference = [TAG_KEY_MAPPING[i] for i in tag_options
                           if i in TAG_KEY_MAPPING]
    column_options = [i for i in df.columns if i in name_col_preference]
    # _logger.debug(f"Column options: {column_options}.")
    return df[column_options].apply(_first_non_null, axis=1)


def identify_user_references(ris_path):
    """Get match table: load references, count citations, match journal names.

    Args:
        ris_path (str): RIS references file path.

    Returns:
        refs_df table (pd.DataFrame). Includes columns for
            - journal matching info (uid, scopus_id, categ, single_match, user_journal)
            - article info (use_article_title, use_year, use_authors),
            - extra article info (may include some/all of type_of_reference,
              primary_title, first_authors, publication_year, alternate_title3,
              volume, number, start_page, end_page, doi, url, and others...)
    """
    _logger.info(f"Loading references file.")
    df = _read_ris_file(ris_path)  # type: pd.DataFrame
    n_null_titles = df.journal.isnull().sum()
    if n_null_titles:
        _logger.info(f"Dropping {n_null_titles} rows with missing journal titles.")
        # if any([pd.isnull(i) for i in journal_names_uniq]):
        #     raise BadRisException("At least one record is missing a journal name.")
        df.dropna(axis='rows', subset=['journal'], inplace=True)
    # Add matching info (uid, categ, single_match) to user refs table
    journal_names_uniq = df.journal.unique()
    from .reference import TM
    m = TM.match_titles(journal_names_uniq)
    _logger.info(f"Matched {m.single_match.sum()} out of {len(m)} cited journals.")
    df = df.join(m.set_index('input_title'), how='left', on='journal')
    # put UID column first
    df = df[['uid'] + [i for i in df.columns if i != 'uid']].copy()
    # rename 'journal' to 'user_journal'
    df.rename(columns={'journal': 'user_journal'}, inplace=True)
    return df


def _first_non_null(title_series):
    titles = title_series.dropna()
    n_options = titles.size
    title = titles.iloc[0] if n_options else np.nan
    return title


def _read_ris_file(ris_data):
    """Load table of references from RIS file path."""
    try:
        if 'seek' in dir(ris_data):
            import codecs
            line = ris_data.readline().encode('utf-8')
            if line.startswith(codecs.BOM_UTF8):
                ris_data.seek(len(codecs.BOM_UTF8))
            else:
                ris_data.seek(0)
            entries = readris(ris_data)
        else:
            with open(ris_data, 'r', encoding='utf-8-sig') as bibfile:
                entries = readris(bibfile)
        df = pd.DataFrame.from_records(entries)
    except Exception as e:
        _logger.error(f"RIS parsing error: {e}")
        raise BadRisException(e)
    _logger.debug(f"Loaded {len(df)} entries via RIS.")
    # RESTRICT TO JOURNAL ROWS
    skipped_types = set(df[_TYPE_COL]).difference(_USE_TYPES)
    if skipped_types:
        _logger.debug(f"Skipped {len(skipped_types)} entry type(s): {skipped_types}")
    df = df[df[_TYPE_COL].isin(_USE_TYPES)]
    # IDENTIFY JOURNAL NAMES
    df.insert(0, 'journal', _get_journal_names(df))
    df.insert(1, 'use_article_title', _get_article_titles(df))
    df.insert(2, 'use_year', _get_years(df))
    df.insert(3, 'use_authors', _get_authors(df))

    # DEDUPLICATE, ignoring all-nan matching fields
    dup_cols = [TAG_KEY_MAPPING[i] for i in _DUP_FIELDS
                if i in TAG_KEY_MAPPING]
    dup_cols = [i for i in dup_cols if i in df.columns]
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
