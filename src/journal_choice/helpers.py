import os
import re
import logging
import unicodedata
from itertools import zip_longest

import numpy as np
import pandas as pd
import pprint
import yaml


_logger = logging.getLogger(__name__)


STRIP_CHARS = ['"', "'", "-", ':', '.', '͡', ',', '[', ']', '=', '?',
               '(', ')', ';', '<', '>', '!', '+', '\\', '|', '@']
# Note: don't bother stripping /, $, &


class IllegalISSNException(Exception):
    pass


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_clean_lowercase(s: str):
    s = ''.join([i for i in s if i not in STRIP_CHARS])
    s = s.casefold()
    normalized = unicodedata.normalize('NFKD', s)
    out = normalized.encode('ascii', errors='ignore').decode('utf8')
    multi_spaces = re.findall(r'\s{2,}', out)
    for space in multi_spaces:
        out = out.replace(space, ' ')
    return out.strip()


def get_issn_safe(issn_series):
    """Get ISSN values with duplicates replaced by nan."""
    illegal_lengths = set(issn_series.apply(len).value_counts().index)\
        .difference({0, 8})
    has_illegal_length = issn_series.apply(len).isin(illegal_lengths)
    bad_inds = list(has_illegal_length.loc[lambda v: v].index)
    bad_ind_vals = set(issn_series.loc[bad_inds].values)
    if bad_inds:
        _logger.warning(f"Bad ISSN at {bad_inds}: {bad_ind_vals}.")
    issn_series = issn_series.where(~has_illegal_length, np.nan)
    dups = (issn_series.value_counts() > 1).loc[lambda v: v].index
    return issn_series.where(~issn_series.isin(dups), np.nan)


def get_issn_comb(print_series, online_series, drop_dups=True):
    """Get combined print + online ISSN string, nan if duplicate.

    Args:
        print_series (pd.Series): print ISSN (without dashes).
        online_series (pd.Series): online ISSN (without dashes).
        drop_dups (bool): use nan if duplicated value.
    """
    issn_comb = map(_get_issn_combined_str, print_series, online_series)
    issn_comb = pd.Series(issn_comb, index=print_series.index)
    issn_lengths = set(issn_comb.apply(len).value_counts().index)
    if issn_lengths.difference({0, 9, 19}):
        raise IllegalISSNException("Illegal ISSN values present")
    if drop_dups:
        comb_dups = (issn_comb.value_counts() > 1).loc[lambda v: v].index
        issn_comb_safe = issn_comb.where(~issn_comb.isin(comb_dups), np.nan)
        return issn_comb_safe
    else:
        return issn_comb


def coerce_issn_to_numeric_string(issn):
    """
    >>> coerce_issn_to_numeric_string('123-45678')
    '12345678'
    >>> coerce_issn_to_numeric_string('004-4586X')
    '0044586X'
    >>> coerce_issn_to_numeric_string('***-*****')
    ''
    """
    if pd.isnull(issn):
        return np.nan
    assert (type(issn) is str), "ISSN must be a string."
    issn = ''.join([i for i in issn if i.isnumeric() or i in {'X', 'x'}])
    return issn.upper()


def _get_issn_combined_str(paper, online):
    """Get ISSN string combining paper and online ISSNs.

    Args:
        paper (str): single paper ISSN value.
        online (str): single online ISSN value (aka E-ISSN).
    """
    values = []
    if paper and not_nan(paper):  # hack to handle nan
        values.append(f'P{paper}')
    if online and not_nan(online):
        values.append(f'E{online}')
    return '_'.join(values)


def not_nan(val):
    if val == val:
        return True
    return False


def get_queries_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('Invalid query YAML path.')
    with open(yaml_path, 'r') as infile:
        query_dict = yaml.load(infile, yaml.SafeLoader)
    _logger.info(f"Inputs:\n{pprint.pformat(query_dict, sort_dicts=False)}.")
    assert not {'title', 'abstract'}.difference(set(query_dict.keys())), \
        "Keys must be title, abstract"
    return query_dict
