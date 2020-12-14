"""
PUBMED journal list at ftp://ftp.ncbi.nih.gov/pubmed/J_Medline.txt
    via https://www.nlm.nih.gov/bsd/serfile_addedinfo.html
"""

import os
import gzip
import time
import pickle
import shlex
import shutil
import logging
import requests
import xmltodict
import contextlib
import multiprocessing
from collections import OrderedDict
from urllib import request as urllib_request

import iso4
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from . import DATA_DIR
from .helpers import get_issn_safe, get_issn_comb, get_clean_lowercase, grouper


JOURNALS_PATH = os.path.join(DATA_DIR, 'J_Medline.txt')
META_PATH = os.path.join(DATA_DIR, 'meta.pickle.gz')
URL_ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
URL_ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
load_dotenv(find_dotenv())  # for NCBI API KEY
ESUMMARY_PATH = os.environ.get('ESUMMARY_PATH')
UID_DICT_PATH = os.path.join(DATA_DIR, 'uid_dict.pickle')
TM = None  # will hold matching functions and data
TM_PICKLE_PATH = os.path.join(DATA_DIR, 'tm.pickle.gz')

_logger = logging.getLogger(__name__)
# _logger.setLevel('DEBUG')


class HTTPError414(Exception):
    pass


class TitleMatcher:
    def __init__(self, pm=None, use_pickle=True):
        """Match user journal title to PubMed journal data.

        Args:
            pm: PubMed reference data table, via e.g. load_pubmed_journals.
        """
        if pm is not None and use_pickle:
            raise ValueError("Specify pm or use_pickle, not both.")

        self.pm = None
        self.titles = None
        self.safe_abbrv_uid_dict = None
        self.safe_uid_dict = None
        self.exact_uid_dict = None
        self.alt_uid_dict = None
        self.alt_safe_uid_dict = None
        self.abbrv_uid_dict = None

        if use_pickle and os.path.exists(TM_PICKLE_PATH):
            _logger.info('loading from TM pickle')
            self._init_from_pickle()
        else:
            self.refresh_matching_data()

    def _init_from_pm_full(self, pmf):
        """Generate matching data from full Pubmed journals table."""
        titles = self._gather_titles(pmf)
        self.titles = titles
        # EXACT title match dictionary
        _logger.info("Building exact title match dictionary.")
        temp = pmf.loc[pmf.is_active, 'main_title'].drop_duplicates(keep=False)
        exact_uid_dict = dict(zip(temp.values, temp.index))
        self.exact_uid_dict = exact_uid_dict
        # ALT title match dictionary
        alt_uid_dict = titles[~titles.canonical].groupby('title')['uid']\
            .apply(lambda s: tuple(s.unique())).to_dict()
        self.alt_uid_dict = alt_uid_dict
        # Safe match dictionary
        _logger.info("Building safe title match dictionary.")
        temp = pmf.loc[pmf.is_active, 'title_safe'].drop_duplicates(keep=False)
        safe_uid_dict = dict(zip(temp.values, temp.index))
        self.safe_uid_dict = safe_uid_dict
        # ALT safe match dictionary
        alt_safe_uid_dict = titles.groupby('title_safe')['uid']\
            .apply(lambda s: tuple(s.unique())).to_dict()
        self.alt_safe_uid_dict = alt_safe_uid_dict
        # Abbreviation dictionaries
        temp = pmf.loc[pmf.is_active, 'MedAbbr'].drop_duplicates(keep=False)
        abbrv_uid_dict = dict(zip(temp.values, temp.index))
        self.abbrv_uid_dict = abbrv_uid_dict
        temp = pmf.loc[pmf.is_active, 'abbr_safe']  # .drop_duplicates(keep=False)
        safe_abbrv_uid_dict = temp.reset_index().groupby('abbr_safe')\
            .apply(lambda g: tuple(g.uid.unique())).to_dict()
        self.safe_abbrv_uid_dict = safe_abbrv_uid_dict

        # TRIM PM: KEEP ESSENTIAL COLUMNS, DROP OTHERS
        rename_dict = {'MedAbbr': 'abbr'}
        keep_cols = ['main_title', 'issn_comb', 'issn_print', 'issn_online',
                     'abbr', 'in_medline']
        pm = pmf.rename(columns=rename_dict)[keep_cols]
        self.pm = pm

    def _init_from_pickle(self):
        with gzip.open(TM_PICKLE_PATH, 'r') as infile:
            data = pickle.load(infile)
        self.safe_abbrv_uid_dict = data['safe_abbrv_uid_dict']
        self.safe_uid_dict = data['safe_uid_dict']
        self.exact_uid_dict = data['exact_uid_dict']
        self.alt_uid_dict = data['alt_uid_dict']
        self.alt_safe_uid_dict = data['alt_safe_uid_dict']
        self.abbrv_uid_dict = data['abbrv_uid_dict']
        self.titles = data['titles']
        self.pm = data['pm']

    def match_titles(self, titles, n_processes=1):
        """Match multiple titles against pubmed sources.

        Args:
            titles (iterable): titles to match.
            n_processes (int): Processor count for mapping single title lookup.
        Returns:
            match table (pd.DataFrame): inc input_title, uid,
                categ (type of match, e.g. safe abbreviation),
                single_match (bool for exactly one match).
        """
        use_multi = n_processes > 1
        if use_multi:
            with multiprocessing.Pool(processes=n_processes) as pool:
                uid_list = list(pool.map(TM.get_uids_from_title, titles))
        else:
            uid_list = list(map(self.get_uids_from_title, titles))
        match = pd.DataFrame.from_records(uid_list, columns=['uid', 'categ'])
        match.insert(0, 'input_title', titles)
        is_unmatched = (match.categ == 'unmatched')
        is_multiple = match.uid.apply(lambda v: type(v) is tuple and len(v) > 1)
        is_single = ~is_unmatched & ~is_multiple  # type: pd.Series
        match['single_match'] = is_single
        if type(titles) is pd.Series:
            match.set_index(titles.index, inplace=True)
        n_nonsingle = (~is_single).sum()
        if n_nonsingle:
            unmatched = list(match.loc[is_unmatched, 'input_title'])
            multiple_match = list(match.loc[is_multiple, 'input_title'])
            _logger.debug(f"Failed to generate unique match for {n_nonsingle} titles: "
                         f"{unmatched=}, {multiple_match=}.")
        else:
            _logger.debug("Uniquely matched all titles")
        return match

    def get_uids_from_title(self, user_title):
        """"Get UIDs corresponding to provided journal title.

        Args:
            user_title (str): Journal name or abbreviation.

        Returns:
            UID string if single UID, otherwise tuple of UIDs.
        """
        # Exact title match
        # PATTERN: query, search_dict, desc

        user_safe = get_clean_lowercase(user_title)
        try:
            user_abbrv = iso4.abbreviate(user_title, periods=True,
                                         disambiguation_langs=['en'])
        except Exception:
            user_abbrv = user_title
        user_abbrv_safe = get_clean_lowercase(user_abbrv)
        # ltwa workaround (journal of biological chemistry > j biol chem)
        user_abbrv_safe = self.tweak_abbreviation(user_abbrv_safe)
        user_safe_the = 'the ' + user_safe
        tests = [
            (user_title, self.exact_uid_dict, 'exact canonical'),
            (user_safe, self.safe_uid_dict, 'exact safe'),
            (user_title, self.abbrv_uid_dict, 'exact abbrv'),
            (user_safe, self.safe_abbrv_uid_dict, 'safe abbrv'),
            (user_title, self.alt_uid_dict, 'exact alt'),
            (user_safe, self.alt_safe_uid_dict, 'safe alt'),
            (user_title, self.abbrv_uid_dict, 'coerced exact abbrv'),
            (user_abbrv_safe, self.safe_abbrv_uid_dict, 'coerced safe abbrv'),
            (user_safe_the, self.safe_uid_dict, 'exact safe THE'),
            (user_safe_the, self.alt_safe_uid_dict, 'safe alt THE'),
        ]
        for query, uid_dict, desc in tests:
            uid = self._try_uid_query(query, uid_dict)
            if uid is not None:
                return uid, desc
        return tuple(), 'unmatched'

    def lookup_title(self, user_str):
        """Helper method to search for string in titles table.

        Returns:
            (pd.DataFrame) table of rows from title table that match title
        """
        user_lower = user_str.lower()
        user_safe = get_clean_lowercase(user_str)
        titles = self.titles
        title_match = titles.title.str.lower().str.contains(user_lower)
        safe_match = titles.title_safe.str.contains(user_safe)
        return titles[title_match | safe_match]

    def _try_uid_query(self, query, uid_dict):
        if query not in uid_dict:
            return
        uids = uid_dict[query]
        if type(uids) is tuple and len(uids) == 1:
            return uids[0]
        return uids

    @staticmethod
    def tweak_abbreviation(user_abbrv):
        """Override abbreviation to handle LTWA problem cases."""
        # ltwa workaround (journal of biological chemistry > j biol chem)
        parts = shlex.split(user_abbrv.lower())
        if len(parts) == 1:
            return user_abbrv
        if 'biological' in parts:
            out_abbrv = user_abbrv.replace('Biological', 'biological')
            out_abbrv = out_abbrv.replace('biological', 'biol')
            return out_abbrv
        return user_abbrv

    @staticmethod
    def _gather_titles(pm):
        title_uids = []
        for uid, r in pm.iterrows():
            if r.is_active:
                main_title = r.titlemainlist[0]['title']
                # These tend to end with '.' for some mysterious reason.
                main_title = main_title.rstrip('.')
                ref_title = r.JournalTitle
                alt_titles = [i['titlealternate'] for i in r['titleotherlist']]
                for title in set([main_title, ref_title] + alt_titles):
                    canonical = True if title == main_title else False
                    title_uids.append((title, uid, canonical))
        titles = pd.DataFrame.from_records(title_uids,
                                           columns=['title', 'uid', 'canonical'])
        titles['title_safe'] = titles['title'].apply(get_clean_lowercase)
        return titles


def load_pubmed_journals(refresh=False):
    """Load table of PubMed journals.

    Reload source from NCBI if refresh is True or file doesn't exist.
    """
    if refresh or not os.path.exists(JOURNALS_PATH):
        refresh_pubmed_reference()
    pm = pd.DataFrame.from_records(_yield_records(JOURNALS_PATH))
    # remove dashes from issn
    pm['ISSN (Print)'] = pm['ISSN (Print)'].str.replace('-', '')
    pm['ISSN (Online)'] = pm['ISSN (Online)'].str.replace('-', '')
    # dup_titles_pubmed = (pm['JournalTitle'].value_counts() > 1)\
    #     .loc[lambda v: v].index
    # pm['is_unique_title'] = ~pm['JournalTitle'].isin(dup_titles_pubmed)
    # get reference issn value (print > online)
    pm['issn_print'] = get_issn_safe(pm['ISSN (Print)'])
    pm['issn_online'] = get_issn_safe(pm['ISSN (Online)'])
    pm['issn_comb'] = get_issn_comb(pm['ISSN (Print)'], pm['ISSN (Online)'])

    _fill_uids(pm, save_pickle=True)
    meta = _load_metadata(pm)
    pm = pm.merge(meta, on='uid', how='inner', validate='one_to_one')
    pm['is_active'] = (pm['endyear'] == '9999')
    pm['abbr_safe'] = pm.MedAbbr.apply(get_clean_lowercase)
    pm.set_index('uid', inplace=True)
    pm.insert(0, 'main_title',
              pm.titlemainlist.apply(lambda v: v[0]['title'].rstrip('.')))
    pm['title_safe'] = pm['main_title'].apply(get_clean_lowercase)
    dup_titles_safe = (pm['title_safe'].value_counts() > 1) \
        .loc[lambda v: v].index
    pm['is_unique_title_safe'] = ~pm['title_safe'].isin(dup_titles_safe)
    pm['in_medline'] = pm['currentindexingstatus'].map({'Y': True, 'N': False})
    return pm


def refresh_pubmed_reference():
    """Download new Pubmed reference (J_Medline.txt) via ftp to data dir."""
    medline_url = "ftp://ftp.ncbi.nih.gov/pubmed/J_Medline.txt"
    with contextlib.closing(urllib_request.urlopen(medline_url)) as r:
        with open(JOURNALS_PATH, 'wb') as f:
            shutil.copyfileobj(r, f)
    _logger.info(f"Updated pubmed reference file from NCBI FTP ({JOURNALS_PATH}).")


def _load_metadata(pm):
    """Load metadata from file, fetching/saving new UIDs if necessary."""
    meta = pd.read_pickle(META_PATH)
    new_uids = set(pm.uid).difference(meta.uid)
    if new_uids:
        # new_names = pm.loc[pm.uid.isin(new_uids), 'JournalTitle']
        meta_new = _build_meta_from_uids(new_uids)
        meta = pd.concat([meta, meta_new], axis=1, ignore_index=True)
        meta.to_pickle(META_PATH)
        _logger.info(f"Update metadata archive with {len(new_uids)} new items.")
    return meta


def _load_uid_dict():
    with open(os.path.join(DATA_DIR, 'uid_dict.pickle'), 'rb') as infile:
        uid_dict = pickle.load(infile)
    return uid_dict


def _fill_uids(pm, save_pickle=True):
    """Use NLM IDs to populate UID column, calling NCBI API if necessary."""
    uid_dict = _load_uid_dict()
    pm['uid'] = _get_uids_from_nlmids(pm['NlmId'], uid_dict=uid_dict)
    nlmids_unknown = list(pm.loc[pm.uid.isnull().loc[lambda v: v].index, 'NlmId'])
    if nlmids_unknown:
        _logger.info(f"Looking up {len(nlmids_unknown)} unrecognized NLM IDs.")
        new_uid_dict = _request_uids_from_nlmids(nlmids_unknown)
        uid_dict.update(new_uid_dict)
        if save_pickle:
            with open(UID_DICT_PATH, 'wb') as outfile:
                pickle.dump(uid_dict, outfile)
                _logger.debug("Updating UID dictionary.")
        for nlmid in nlmids_unknown:
            pm.loc[pm.NlmId == nlmid, 'uid'] = new_uid_dict[nlmid]


def _get_uids_from_nlmids(nlmid_series, uid_dict=None):
    if uid_dict is None:
        uid_dict = {}
    uids = nlmid_series.map(uid_dict)  # null for those not in dictionary
    is_numeric = nlmid_series.str.isnumeric()
    zero_start = nlmid_series.str.startswith('0')
    uids = nlmid_series.where(is_numeric & ~zero_start, uids)
    uids = nlmid_series.str.lstrip('0').where(zero_start & is_numeric, uids)
    return uids


def _build_meta_from_uids(uids, batch_size=400):
    """Build metadata table for NLM UIDs from NCBI esummary API.

    Returns:
        meta (pd.DataFrame): table of journal/book extended metadata.
    """
    # iterate 500 uids at a time
    n_batches = int(np.ceil(len(uids) / batch_size))
    ind = 0
    all_records = []
    for ind_g, g in enumerate(grouper(uids, batch_size)):
        ind += 1
        uids_batch = [i for i in g if i]
        try:
            records = _get_meta_records_from_ids(uids_batch)
        except HTTPError414 as err:
            _logger.info(f"414 at index {ind} ({err}), splitting into two requests.")
            n_half = round(len(uids_batch) / 2)
            uids1 = uids_batch[:n_half]
            uids2 = uids_batch[n_half:]
            records1 = _get_meta_records_from_ids(uids1)
            records2 = _get_meta_records_from_ids(uids2)
            records = records1 + records2
            ind += 1
        _logger.info(f"Progress: finished lookup for batch {ind_g} of {n_batches}")
        all_records.extend(records)
    meta = pd.DataFrame.from_records(all_records)
    return meta


def _get_meta_records_from_ids(uids_batch):
    collapse_list_vars = [
        'authorlist',
        'publicationinfolist',
        'resourceinfolist',
        # 'titlemainlist',
        # 'titleotherlist',
        'issnlist']
    records = []  # will hold all meta records
    ids = ','.join(uids_batch)
    # CF: esummary -db nlmcatalog -mode json -id {ids} > group_uids.txt
    res = requests.post(URL_ESUMMARY,
                        params=dict(db='nlmcatalog', id=ids, retmode='json',
                                    api_key=os.environ['API_KEY']))
    # test for res.status_code == 414. failure at 4154
    if res.status_code == 414:
        raise HTTPError414(f"URL too long: {len(res.url)} characters.")
    j = res.json()
    uids_response = j['result']['uids']
    for uid in uids_response:
        new_dict = OrderedDict()
        v = j['result'][uid]
        for i in v.keys():
            if i not in collapse_list_vars:
                new_dict[i] = v[i]
            elif v[i]:
                first_dict = v[i][0]
                new_dict.update(first_dict)
        records.append(new_dict)
    return records


# nlmid -> uid: strip numeric or lookup for A-Z inclusion


def _yield_records(journals_path):
    """Helper for loading PubMed journals file."""
    with open(journals_path, 'r') as infile:
        infile.readline()  # skip first '---' line
        record = OrderedDict()
        for line in infile:
            if line.startswith('-') or not line:
                yield record
                record = OrderedDict()
                continue
            vals = line.strip().split(':')
            field = vals[0].strip()
            val = ''.join(vals[1:]).strip()
            record[field] = val
        yield record


def _request_uids_from_nlmids(nlmids):
    """Use eSearch API to get NLM UIDs from NLM IDs.

    Args:
        nlmids (iterable): NLM IDs as strings.

    Returns:
        dictionary of NLM ID: NLM UID
    """
    uid_dict = dict()
    for ind, nlmid in enumerate(nlmids):
        res = requests.get(URL_ESEARCH,
                           params=dict(db='nlmcatalog', id=nlmid,
                                       term=nlmid, field='nlmid',
                                       api_key=os.environ['API_KEY']))
        out = xmltodict.parse(res.content)['eSearchResult']
        if out['Count'] != '1':
            _logger.error(f'Non single result for {nlmid}')
        else:
            uid = out['IdList']['Id']
            uid_dict[nlmid] = uid
        if not ind % 100:
            _logger.debug(f"Download progress: finished lookup for index {ind}")
    return uid_dict


def _unused_get_failed_lookup_ids(script_path, problem_output_paths):
    lines = []
    with open(script_path, 'r') as infile:
        for line in infile:
            for path in problem_output_paths:
                if path in line:
                    lines.append(line)
    problem_ids = []
    for line in lines:
        ids = line.split('-id ')[1].split(' > ')[0].split(',')
        problem_ids.extend(_unused_test_nlmids_for_lookup_failure(ids))
    return problem_ids


def _unused_test_nlmids_for_lookup_failure(ids):
    problem_ids = []
    for nlmid in ids:
        res = requests.get(URL_ESUMMARY, params=dict(db='nlmcatalog', id=nlmid))
        if 'ERROR' in res.content.decode('utf8'):
            _logger.info(f"Problem ID: {nlmid}")
            problem_ids.append(nlmid)
        time.sleep(0.1)
    if not problem_ids:
        _logger.info("IDs looked up successfully.")
    return problem_ids


def load_scopus_map():
    """Get dictionary of NLM UID -> Scopus ID, based on build_uid_match_table."""
    with open(MATCH_JSON_PATH) as infile:
        scopus_id_dict = json.load(infile)
    return scopus_id_dict


TM = TitleMatcher()
