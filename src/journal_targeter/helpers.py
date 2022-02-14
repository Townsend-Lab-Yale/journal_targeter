import os
import re
import json
import glob
import gzip
import pickle
import logging
import hashlib
import pathlib
import unicodedata
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
import pandas as pd
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
    issn_series = issn_series.str.replace('-', '').copy()
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


def coerce_to_valid_issn_or_nan(issn):
    """Gives 8-length valid ISSN string or nan.

    >>> coerce_to_valid_issn_or_nan('123-45678')
    '12345678'
    >>> coerce_to_valid_issn_or_nan('004-4586X')
    '0044586X'
    >>> _logger.setLevel("ERROR")
    >>> coerce_to_valid_issn_or_nan('1234567Y')
    nan
    >>> coerce_to_valid_issn_or_nan('***-*****')
    nan
    >>> coerce_to_valid_issn_or_nan('')
    nan
    """
    if pd.isnull(issn) or issn == '':
        return np.nan
    assert (type(issn) is str), "ISSN must be a string."
    new_issn = ''.join([i for i in issn if i.isnumeric() or i in {'X', 'x'}])
    new_issn = new_issn.upper()
    if len(new_issn) != 8:
        _logger.info(f"Skipping illegal issn: {issn=}, (coerced={new_issn}).")
        return np.nan
    return new_issn


def _get_issn_combined_str(paper, online):
    """Get ISSN string combining paper and online ISSNs.

    Args:
        paper (str): single paper ISSN value.
        online (str): single online ISSN value (aka E-ISSN).

    >>> _get_issn_combined_str('12345678', '87654321')
    'P12345678_E87654321'
    >>> _get_issn_combined_str(np.nan, '12345678')
    'E12345678'
    >>> _get_issn_combined_str('', np.nan)
    ''
    """
    values = []
    if paper and pd.notna(paper):  # hack to handle nan
        values.append(f'P{paper}')
    if online and pd.notna(online):
        values.append(f'E{online}')
    return '_'.join(values)


def get_queries_from_yaml(yaml_input):
    """Load title, abstract dictionary from yaml path or string."""
    if os.path.exists(yaml_input):
        with open(yaml_input, 'r') as infile:
            query_dict = yaml.load(infile, yaml.SafeLoader)
    else:
        query_dict = yaml.load(yaml_input, yaml.SafeLoader)
    _logger.info(f"Inputs:\n{query_dict}.")
    assert not {'title', 'abstract'}.difference(set(query_dict.keys())), \
        "Keys must be title, abstract"
    return query_dict


def concat_json_data_records_as_single_file(glob_str, out_path):
    paths = sorted([pathlib.Path(i) for i in glob.glob(glob_str)],
                   key=lambda p: p.stat().st_ctime, reverse=True)
    data = []
    for path in paths:
        with open(path, 'rb') as json_in:
            data.extend(json.load(json_in)['data'])
    n_records = len(data)
    with open(out_path, 'w') as out:
        json.dump({'data': data}, out, ensure_ascii=False)
    _logger.info(f"{n_records} records written to '{out_path}'")


def load_jcr_json(json_path, name_col='journalName',
                  print_issn_col='issn', online_issn_col='eissn',
                  jif_col='jif2019', jci_col='jci',
                  ai_col='articleInfluenceScore', ef_col='eigenFactor',
                  efn_col='normalizedEigenFactor', nan_val='n/a',
                  issn_placeholder='****-****'):
    """Create dataframe from JCR json files specified by glob string."""
    with open(json_path, 'r') as infile:
        jcr = json.load(infile)
    df = pd.DataFrame.from_records(jcr['data'])
    df = df.applymap(lambda v: np.nan if v == nan_val else v)
    df[print_issn_col].replace(issn_placeholder, '', inplace=True)
    df[online_issn_col].replace(issn_placeholder, '', inplace=True)
    keep_cols = OrderedDict({
        name_col: 'journal',
        print_issn_col: 'issn_print',
        online_issn_col: 'issn_online',
        jif_col: 'JIF',
        jci_col: 'JCI',
        ai_col: 'AI',
        ef_col: 'EF',
        efn_col: 'EFn',
    })
    df_sm = df[keep_cols].rename(columns=keep_cols)
    assert (df_sm['journal'].value_counts().max() == 1), "Some journal names are duplicated."
    metric_cols = ['JIF', 'JCI', 'AI', 'EF', 'EFn']
    for col in metric_cols:
        df_sm[col] = df_sm[col].astype(float)
    return df_sm


def pickle_seems_ok(pickle_path):
    is_gzipped = pickle_path.endswith('.gz')
    open_fn = gzip.open if is_gzipped else open
    if not os.path.exists(pickle_path):
        return False
    with open_fn(pickle_path, 'rb') as infile:
        try:
            temp = pickle.load(infile)
            return True
        except:
            return False


def get_md5(file_path):
    h = hashlib.md5()
    with open(file_path, 'rb') as infile:
        for line in infile:
            h.update(line)
    return h.hexdigest()
