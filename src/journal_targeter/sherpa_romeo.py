import os
import gzip
import glob
import pickle
import logging
from collections import namedtuple
from typing import List, Optional

import requests
import numpy as np
import pandas as pd

from . import paths


FIELD_NAMES = ['sr_id', 'title', 'issn_print', 'issn_online', 'issn_other']
Journal = namedtuple('Journal', field_names=FIELD_NAMES)
_logger = logging.getLogger(__name__)


def download_sherpa_data(api_key: Optional[str], delete_old: bool = False) -> None:
    """Save Sherpa Romeo records in batches of 100 to download_dir as pickle.

    - If a downloaded file already exists, it is skipped.
    - Takes approx 35min to download one at a time. Could be parallelized if
      number of journals is known in advance.
    """
    api_key = os.environ.get('ROMEO_KEY') if not api_key else api_key
    if not api_key:
        raise MissingKeyException("Supply api_key argument or set ROMEO_KEY "
                                  "environment variable")
    for offset in range(0, 100000, 100):  # assume <100k. 32185 on 2021-08-05
        if not offset % 1000:
            _logger.info(f"Download progress: current record index = {offset}.")
        out_path = os.path.join(paths.ROMEO_TMP, f"items_{offset:05.0f}.pickle.gz")
        if os.path.exists(out_path):
            continue
        r = requests.get('https://v2.sherpa.ac.uk/cgi/retrieve',
                         params={'item-type': 'publication',
                                 'api-key': api_key,
                                 'format': 'Json', 'limit': 100,
                                 'offset': offset})
        data = r.json()
        res_list = data['items']
        n_items = len(res_list)
        if n_items == 0:
            break
        with gzip.open(out_path, 'wb') as out:
            pickle.dump(res_list, out)  # save results


def delete_old_sherpa_data():
    old_paths = glob.glob(os.path.join(paths.ROMEO_TMP, 'items_*.pickle.gz'))
    for path in old_paths:
        os.remove(path)
    _logger.info(f"Removed {len(old_paths)} old Sherpa Romeo files.")


def match_sherpa_titles_issns(download_dir: str,
                              n_processes: Optional[int] = None) -> pd.DataFrame:
    """Get NLM IDs from Sherpa Romeo titles and ISSNs. Slow!

    Args:
        download_dir (str): Directory path with Sherpa Romeo pickled data.
        n_processes (int): Number of processes for matching. Fallback to
            N_PROCESSES in environment if exists, else 1.

    Returns:
        Table of SR data with nlmids. Columns are title, issn_print, issn_online,
            issn_other, issn_other1, issn_print_merge, nlmid.
    """
    if not n_processes:
        n_processes = os.environ.get("N_PROCESSES", 1)
    _logger.info(f"Matching Sherpa Romeo data using {n_processes} processes.")
    j_list = _get_records_from_sherpa_downloads(download_dir)  # 10s to load files
    sr = pd.DataFrame.from_records(j_list, columns=FIELD_NAMES)
    sr.set_index('sr_id', inplace=True)
    sr['issn_other1'] = sr['issn_other'].map(
        lambda v: v.split('|')[0] if type(v) is str else np.nan)
    sr['issn_print_merge'] = sr['issn_print'].where(pd.notnull, sr['issn_other1'])
    # 8min 30s with 4 processes. 12571 matched, 19614 unmatched. 1 invalid ISSN.
    from .reference import TM
    matched = TM.lookup_uids_from_title_issn(sr['title'], sr['issn_print_merge'],
                                             sr['issn_online'],
                                             n_processes=n_processes)
    # matched.to_pickle('sr_matched.pickle.gz')
    sr['nlmid'] = matched['uid'].values
    return sr


def save_sherpa_id_map(sr: pd.DataFrame) -> pd.DataFrame:
    """Save Sherpa Romeo ID -> NLM ID map as table.

    Args:
        sr: Dataframe output of match_sherpa_titles_issns.
        out_dir: Directory in which to save sherpa_romeo_map.tsv.gz

    Returns:
        Table with columns sr_id, nlmid.
    """
    sr_map = sr.dropna(subset=['nlmid']).reset_index()[['sr_id', 'nlmid']]
    out_path = paths.ROMEO_MAP_PATH
    sr_map.to_csv(out_path, sep='\t', compression='gzip',
                  index=False)
    _logger.info(f"Saved Sherpa Romeo ID mapping data to '{out_path}'.")
    return sr_map


def load_sherpa_id_map() -> pd.DataFrame:
    """Save Sherpa Romeo ID -> NLM ID map as table.

    Returns:
        Table with columns sr_id, nlmid.
    """
    map_path = paths.ROMEO_MAP_PATH
    sr_map = pd.read_csv(map_path, sep='\t', compression='gzip', dtype=str)
    sr_map = sr_map.set_index('nlmid')['sr_id']
    return sr_map


def _get_records_from_sherpa_downloads(download_dir) -> List:
    paths = sorted(glob.glob(os.path.join(download_dir, '*.pickle.gz')))
    j_list = []
    for path in paths:
        with gzip.open(path, 'rb') as infile:
            res_list = pickle.load(infile)
            for item in res_list:
                title = item['title'][0]['title']
                sr_id = item['id']
                issn_print, issn_online, issn_other = np.nan, np.nan, np.nan
                if 'issns' in item:
                    for issn_dict in item['issns']:
                        if 'issn' not in issn_dict:
                            continue
                        if 'type' not in issn_dict:
                            issn_other = issn_dict['issn'] if pd.isna(issn_other) \
                                else '|'.join([issn_other, issn_dict['issn']])
                        elif issn_dict['type'] == 'print':
                            issn_print = issn_dict['issn']
                        elif issn_dict['type'] == 'electronic':
                            issn_online = issn_dict['issn']
                j = Journal(sr_id=sr_id,
                            title=title,
                            issn_print=issn_print,
                            issn_online=issn_online,
                            issn_other=issn_other,)
                j_list.append(j)
    return j_list


class MissingKeyException(Exception):
    pass
