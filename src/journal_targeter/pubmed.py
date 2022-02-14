"""Manage Pubmed journals list and metadata, for use as a master journal list.

PUBMED journal list at ftp://ftp.ncbi.nih.gov/pubmed/J_Medline.txt
    via https://www.nlm.nih.gov/bsd/serfile_addedinfo.html

Example record::

    --------------------------------------------------------
    JrId: 1
    JournalTitle: AADE editors' journal
    MedAbbr: AADE Ed J
    ISSN (Print): 0160-6999
    ISSN (Online):
    IsoAbbr: AADE Ed J
    NlmId: 7708172
    --------------------------------------------------------

There are two IDs provided per record. Strangely, neither of these is the UID
used by entrez, which is in the journal page URL in the NLM catalog::

    https://www.ncbi.nlm.nih.gov/nlmcatalog/<UID>


"""

import os
import gzip
import time
import shlex
import shutil
import pickle
import logging
import requests
import xmltodict
import contextlib
import multiprocessing
from collections import defaultdict
from typing import Iterable, Union
from collections import OrderedDict
from urllib import request as urllib_request
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


import iso4
import numpy as np
import pandas as pd
from flask import current_app

from . import paths
from .helpers import get_issn_safe, get_issn_comb, get_clean_lowercase, grouper, \
    coerce_to_valid_issn_or_nan, get_md5, pickle_seems_ok
from .app.models import Source


URL_ESUMMARY = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
URL_ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
_logger = logging.getLogger(__name__)


_retry_strategy = Retry(
    total=3,
    status_forcelist=[413, 429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    backoff_factor=1,
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
http = requests.Session()
http.mount("https://", _adapter)
http.mount("http://", _adapter)


class HTTPError414(Exception):
    pass


class DataNotAvailable(Exception):
    pass


class TitleMatcher:
    """Store NLM/pubmed metadata and dictionaries that map title variants -> UID.

    Attributes:
        titles: table of uids + title variants

        exact_id_dict: mixed case 'main' title -> uid
        alt_id_dict:  mixed case 'alternative' title -> uid
        alt_safe_id_dict: simplified alt title -> uid
        abbrv_id_dict: title abbreviations -> uid
        safe_abbrv_id_dict: simplified lowercase abbreviations -> uid
        issnp_unique: Print ISSN table (nlmid, issn_print)
        issno_unique: Online ISSN table (nlmid, issn_online)
        issnc_unique: Combined Print+Online ISSN table (nlmid, issn_comb)
    """
    def __init__(self):
        self.titles = None
        self.safe_abbrv_id_dict = None
        self.safe_id_dict = None
        self.exact_id_dict = None
        self.alt_id_dict = None
        self.alt_safe_id_dict = None
        self.abbrv_id_dict = None
        self.issnp_unique = None
        self.issno_unique = None
        self.issnc_unique = None
        self.populated = False

    def init_data(self, pm: Union[pd.DataFrame, None] = None):
        if pm is not None:
            _logger.info("Building TitleMatcher dictionaries from pubmed table.")
            self._init_from_pm_full(pm)
            self.populated = True
            self.save_pickle()
        elif os.path.exists(paths.TM_PICKLE_PATH) and \
                pickle_seems_ok(paths.TM_PICKLE_PATH):
            _logger.debug('Loading from TM pickle.')
            self._init_from_pickle()
        else:
            _logger.info("Building TitleMatcher from initial pubmed data.")
            pm = load_pubmed_journals()
            self._init_from_pm_full(pm)
            self.populated = True
            self.save_pickle()
        return self

    def save_pickle(self):
        if not self.populated:
            _logger.error("Attempted to save TM pickle from unpopulated object.")
            return False
        self._write_pickle()

    def _init_from_pm_full(self, pmf):
        """Generate matching data from full Pubmed journals table.

        Args:
            pmf (pd.DataFrame): journals table. Required columns:
                main_title, JournalTitle, alt_titles_str, issn_print,
                issn_online, abbr
            save_file (bool): write data to file for fast loading in future
        """
        pm = pmf.copy()
        pm['issn_comb'] = get_issn_comb(pm['issn_print'], pm['issn_online'])
        pm['abbr_safe'] = pm['abbr'].apply(get_clean_lowercase)
        pm['title_safe'] = pm['main_title'].apply(get_clean_lowercase)

        self.issnp_unique = pm['issn_print'].dropna().drop_duplicates(
            keep=False).reset_index()  # print issn uniquely points to uid
        self.issno_unique = pm['issn_online'].dropna().drop_duplicates(
            keep=False).reset_index()  # online issn uniquely points to uid
        self.issnc_unique = pm['issn_comb'].dropna().drop_duplicates(
            keep=False).reset_index()

        titles = self._gather_titles(pm)
        self.titles = titles
        # EXACT title match dictionary
        _logger.info("Building exact title match dictionary.")
        temp = pm['main_title'].drop_duplicates(keep=False)
        exact_id_dict = dict(zip(temp.values, temp.index))
        self.exact_id_dict = exact_id_dict
        # ALT title match dictionary
        alt_id_dict = titles[~titles.canonical].groupby('title')['nlmid']\
            .apply(lambda s: tuple(s.unique())).to_dict()
        self.alt_id_dict = alt_id_dict
        # Safe match dictionary
        _logger.info("Building safe title match dictionary.")
        temp = pm['title_safe'].drop_duplicates(keep=False)
        safe_id_dict = dict(zip(temp.values, temp.index))
        self.safe_id_dict = safe_id_dict
        # ALT safe match dictionary
        _logger.info("Building alternative safe title match dictionary.")
        alt_safe_id_dict = titles.groupby('title_safe')['nlmid']\
            .apply(lambda s: tuple(s.unique())).to_dict()
        self.alt_safe_id_dict = alt_safe_id_dict
        # Abbreviation dictionaries
        _logger.info("Building abbreviation dictionaries.")
        temp = pm['abbr'].drop_duplicates(keep=False)
        abbrv_uid_dict = dict(zip(temp.values, temp.index))
        self.abbrv_id_dict = abbrv_uid_dict
        temp = pm['abbr_safe']  # .drop_duplicates(keep=False)
        safe_abbrv_id_dict = temp.reset_index().groupby('abbr_safe')\
            .apply(lambda g: tuple(g.nlmid.unique())).to_dict()
        self.safe_abbrv_id_dict = safe_abbrv_id_dict

    def _init_from_pickle(self):
        with gzip.open(paths.TM_PICKLE_PATH, 'r') as infile:
            data = pickle.load(infile)
        self.safe_abbrv_id_dict = data['safe_abbrv_id_dict']
        self.safe_id_dict = data['safe_id_dict']
        self.exact_id_dict = data['exact_id_dict']
        self.alt_id_dict = data['alt_id_dict']
        self.alt_safe_id_dict = data['alt_safe_id_dict']
        self.abbrv_id_dict = data['abbrv_id_dict']
        self.titles = data['titles']
        self.issnp_unique = data['issnp_unique']
        self.issno_unique = data['issno_unique']
        self.issnc_unique = data['issnc_unique']

    def match_titles(self, titles, n_processes=1):
        """Match multiple titles against pubmed sources.

        Args:
            titles (iterable): titles to match.
            n_processes (int): Processor count for mapping single title lookup.
        Returns:
            match table (pd.DataFrame), with columns input_title, uid,
                categ (type of match, e.g. safe abbreviation),
                single_match (bool for exactly one match).
        """
        use_multi = n_processes > 1
        if use_multi:
            with multiprocessing.Pool(processes=n_processes) as pool:
                uid_list = list(pool.map(self.get_uids_from_title, titles))
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
            _logger.debug("Uniquely matched all titles.")
        return match

    def get_uids_from_title(self, user_title):
        """Get UIDs corresponding to provided journal title.

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
        user_abbrv_safe = self._tweak_abbreviation(user_abbrv_safe)
        user_safe_the = 'the ' + user_safe
        tests = [
            (user_title, self.exact_id_dict, 'exact canonical'),
            (user_safe, self.safe_id_dict, 'exact safe'),
            (user_title, self.abbrv_id_dict, 'exact abbrv'),
            (user_safe, self.safe_abbrv_id_dict, 'safe abbrv'),
            (user_title, self.alt_id_dict, 'exact alt'),
            (user_safe, self.alt_safe_id_dict, 'safe alt'),
            (user_title, self.abbrv_id_dict, 'coerced exact abbrv'),
            (user_abbrv_safe, self.safe_abbrv_id_dict, 'coerced safe abbrv'),
            (user_safe_the, self.safe_id_dict, 'exact safe THE'),
            (user_safe_the, self.alt_safe_id_dict, 'safe alt THE'),
        ]
        for query, uid_dict, desc in tests:
            uid = self._try_uid_query(query, uid_dict)
            if uid is not None:
                return uid, desc
        return tuple(), 'unmatched'

    def lookup_uids_from_title_issn(self, titles=None,
                                    issn_print=None,
                                    issn_online=None,
                                    n_processes=1,
                                    resolve_uid_conflicts=True):
        """Look up NLM UIDs using journal names and ISSNs, if provided.

        Matching priority: ISSN print+online > ISSN print > ISSN online > title.

        Args:
            titles (iterable): journal names to be matched.
            issn_print (iterable): OPTIONAL print or generic ISSN iterable.
            issn_online (iterable): OPTIONAL online ISSN / e-ISSN iterable.
            n_processes (int): number of processes. Use mutiprocessing if >1.
            resolve_uid_conflicts (bool): if True, disallow multiple records to map
                to same UID, choosing winner based on matching priority/score.
        Returns:
            match (pd.DataFrame), including `input_title`, `uid`, and other matching
            metadata columns. Discrepant UIDs from multiple sources are indicated by
            `n_vals` > 1, with sources separated by '|'.
        """
        time_start = time.perf_counter()

        match = self.match_titles(list(titles), n_processes=n_processes)
        match.rename(columns={
            'uid': 'on_name',
            'categ': 'name_method',
            'single_match': 'on_name_single',
        }, inplace=True)
        match['bool_name'] = match.name_method != 'unmatched'
        if issn_print is None:
            # NO ISSN DATA, SO RETURN MATCHES
            match['uid'] = match.on_name.where(match.on_name_single, np.nan)
            return match

        # ISSN reference for print and online
        issn_dict = self.match_issns(issn_print=issn_print, issn_online=issn_online)
        for col in issn_dict:
            match[col] = issn_dict[col]
        has_comb_issn = 'on_issnc' in match.columns
        # Add boolean columns for ISSN variant matches
        if has_comb_issn:
            match['bool_issnc'] = ~match.on_issnc.isnull()
            match['bool_issno'] = ~match.on_issno.isnull()
        match['bool_issnp'] = ~match.on_issnp.isnull()

        # Count matches and categorize discrepancies for each scopus ID
        var_cols = ('issnc', 'issnp', 'issno') if has_comb_issn else ('issnp',)
        categs = pd.DataFrame.from_records(
            match.apply(lambda r: self._classify_ids(r, cols=var_cols), axis=1).values,
            columns=['n_vals', 'categ'], index=match.index)
        match = pd.concat([match, categs], axis=1)

        # Resolve competing UIDs with ISSN combined > ISSN print > ISSN online > title
        if has_comb_issn:
            winner = match.on_issnp.where(match.on_issnc.isnull(), match.on_issnc)
            winner = match.on_issno.where(winner.isnull(), winner)
        else:
            winner = match.on_issnp
        winner = match.on_name.where(match.on_name_single & winner.isnull(), winner)
        match['uid'] = winner

        # Handle cases where multiple source IDs map to single NLM UID.
        ambig_uids = set((match['uid'].value_counts() > 1).loc[lambda v: v].index.values)
        if ambig_uids and resolve_uid_conflicts:
            conflict_uids = set((match['uid'].value_counts() > 1).loc[lambda v: v].index.values)
            conflicts = match[match['uid'].isin(conflict_uids)].reset_index()
            conflicts['score'] = conflicts.apply(self._get_match_score, axis=1)
            drop_indices = set()
            match_index_name = match.index.name or 'index'
            for uid, g in conflicts.groupby('uid'):
                max_score = g.score.max()
                is_max = g.score.eq(max_score)
                if is_max.sum() == 1:
                    drop_indices.update(g.loc[~is_max, match_index_name])
                else:
                    drop_indices.update(g[match_index_name])
            match.loc[drop_indices, 'uid'] = np.nan
            match['dropped'] = match.index.isin(drop_indices)
            n_dropped = len(drop_indices)
            _logger.info(f"Dropped {n_dropped} matches during conflict resolution.")
            assert match.uid.value_counts().max() in [1, np.nan], \
                "Scopus matching still includes conflicts."
        elif ambig_uids and not resolve_uid_conflicts:
            _logger.info(f"Note: {len(ambig_uids)} records map to same UID.")
        elif not ambig_uids:
            _logger.info("No conflicting matches found when mapping to pubmed.")
        time_end = time.perf_counter()
        n_seconds = time_end - time_start
        _logger.info(f"Matching to Pubmed UIDs took {n_seconds/ 60:.0f}m{n_seconds % 60:.0f}s.")
        n_unmatched = match.uid.isnull().sum()
        n_matched = len(match) - n_unmatched
        _logger.info(f"Successfully matched {n_matched} journals, leaving {n_unmatched} "
                     f"not linked to pubmed.")
        return match

    def match_issns(self, issn_print: pd.Series = None, issn_online: pd.Series = None):
        # ISSN MATCHING
        issn_dict = OrderedDict()
        df = self._get_issn_query_table(issn_print=issn_print, issn_online=issn_online)
        if 'issn_comb' in df.columns:
            # issn_comb matching useful when issnp repeated but issnp+issno seen once
            df_issn_comb = df['issn_comb'].drop_duplicates(keep=False).reset_index()
            issnc = self.issnc_unique.merge(df_issn_comb, how='inner', on='issn_comb') \
                .set_index('index')['nlmid']
            issn_dict['on_issnc'] = issnc
            # Add Print ISSN matching info
            df_issnp = df['issn_print'].dropna().drop_duplicates(keep=False).reset_index()
            issnp = self.issnp_unique.merge(df_issnp, how='inner', on='issn_print').set_index('index')['nlmid']
            issn_dict['on_issnp'] = issnp
            # Add Online ISSN matching info
            df_issno = df['issn_online'].dropna().drop_duplicates(keep=False).reset_index()
            issno = self.issno_unique.merge(df_issno, how='inner', on='issn_online').set_index('index')['nlmid']
            issn_dict['on_issno'] = issno
        else:  # single issn provided per title
            issns = pd.DataFrame({
                'uid': (list(self.issnp_unique['nlmid']) +
                        list(self.issno_unique['nlmid'])),
                'issn': (list(self.issnp_unique['issn_print']) +
                         list(self.issno_unique['issn_online'])),
                'categ': (list(np.repeat('print', len(self.issnp_unique))) +
                          list(np.repeat('online', len(self.issno_unique)))),
            })
            # all issn-uid pairs
            issns = issns.groupby(['uid', 'issn']).aggregate(lambda s: ','.join(set(s))).reset_index()
            # remove issns that are in more than one pair
            ambig_issns = issns.issn.value_counts().loc[lambda v: v > 1].index
            issns1 = issns[~issns['issn'].isin(ambig_issns)]
            single_issn_dict = issns1.set_index('issn')['uid'].to_dict()
            issn_dict['on_issnp'] = df['issn_print'].map(single_issn_dict)
        return issn_dict

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

    @staticmethod
    def _get_issn_query_table(issn_print=None, issn_online=None):
        # BUILD QUERY DATAFRAME
        issn_print = [coerce_to_valid_issn_or_nan(i) for i in issn_print]
        data = {'issn_print': issn_print}
        has_issn_online = issn_online is not None
        if has_issn_online:
            issn_online = [coerce_to_valid_issn_or_nan(i) for i in issn_online]
            data.update({'issn_online': issn_online})
            issn_comb = get_issn_comb(pd.Series(issn_print), pd.Series(issn_online))
            data.update({'issn_comb': issn_comb})
        df = pd.DataFrame(data)
        return df

    def _classify_ids(self, r, cols=('issnc', 'issnp', 'issno')):
        """Count UIDs and create 'category' string to describe sources and discrepancies.

        A vertical bar (|) separates discrepant sources of UID.
        e.g. categ='issno|issnp|title' means the UID resulting from matching on
        a) issn online, b) issn print, and c) journal title, ALL DISAGREE.
        """
        vals = set()
        d = dict()  # var: uid
        if r.on_name_single:
            vals.add(r.on_name)
            d['title'] = r.on_name
        for var in cols:
            bool_col, id_col = f"bool_{var}", f"on_{var}"
            if r[bool_col]:
                vals.add(r[id_col])
                d[var] = r[id_col]
        n_vals = len(vals)
        categ = self._get_categ_from_uid_dict(d)
        return n_vals, categ

    def _get_categ_from_uid_dict(self, d):
        dd = defaultdict(list)
        for i in d:
            dd[d[i]].append(i)
        categ = '|'.join(sorted(['_'.join(sorted(i)) for i in dd.values()]))
        return categ

    @staticmethod
    def _try_uid_query(query, uid_dict):
        if query not in uid_dict:
            return
        uids = uid_dict[query]
        if type(uids) is tuple and len(uids) == 1:
            return uids[0]
        return uids

    @staticmethod
    def _tweak_abbreviation(user_abbrv):
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
        title_ids = []
        for nlmid, r in pm.iterrows():
            # main_title tends to end with '.'.
            main_title = r['main_title']
            ref_title = r['JournalTitle']
            alt_titles = [i for i in r['alt_titles_str'].split('|') if i]
            for title in set([main_title, ref_title] + alt_titles):
                canonical = True if title == main_title else False
                title_ids.append((title, nlmid, canonical))
        titles = pd.DataFrame.from_records(title_ids,
                                           columns=['title', 'nlmid', 'canonical'])
        titles['title_safe'] = titles['title'].apply(get_clean_lowercase)
        return titles

    def _write_pickle(self):
        data = dict()
        data['safe_abbrv_id_dict'] = self.safe_abbrv_id_dict
        data['safe_id_dict'] = self.safe_id_dict
        data['exact_id_dict'] = self.exact_id_dict
        data['alt_id_dict'] = self.alt_id_dict
        data['alt_safe_id_dict'] = self.alt_safe_id_dict
        data['abbrv_id_dict'] = self.abbrv_id_dict
        data['titles'] = self.titles
        data['issnp_unique'] = self.issnp_unique
        data['issno_unique'] = self.issno_unique
        data['issnc_unique'] = self.issnc_unique
        with gzip.open(paths.TM_PICKLE_PATH, 'w') as infile:
            pickle.dump(data, infile)
        _logger.info("TitleMatcher data written to pickle.")

    @staticmethod
    def _get_match_score(r: pd.Series):
        score = 0
        if 'issn' in r.categ:
            score += 8
        if r.name_method == 'exact canonical':
            score += 4
        elif r.name_method == 'exact safe':
            score += 2
        elif r.name_method != 'unmatched':
            score += 1
        return score


def load_pubmed_journals(api_key: Union[str, None] = None):
    """Load table of PubMed journals, append extended metadata, drop inactive.

    Reload source from NCBI if file doesn't exist. Within application context,
    update time will be stored in db source table.

    Args:
        api_key: NCBI Entrez API Key (optional). Fallback priority: API_KEY in
            current_app.config, then environment.
    """
    if api_key is None:
        if current_app:
            api_key = current_app.config['API_KEY']
        else:
            api_key = os.environ.get('API_KEY', None)
    if not os.path.exists(paths.PM_MEDLINE_PATH):
        _download_pubmed_reference()
    pm = pd.DataFrame.from_records(_yield_records(paths.PM_MEDLINE_PATH))
    meta, meta_changed = _load_metadata_for_nlmids(pm.nlmid, drop_extra=True,
                                                   api_key=api_key)
    if meta_changed and current_app:
        Source.updated_now('pubmed')
    pm.set_index('nlmid', inplace=True)
    pm = pm.join(meta[meta.is_active], how='inner')  # reduces pm to active only
    pm.rename(columns={'MedAbbr': 'abbr'}, inplace=True)
    pm['issn_print'] = get_issn_safe(pm['ISSN (Print)'])
    pm['issn_online'] = get_issn_safe(pm['ISSN (Online)'])
    pm.drop(['JrId', 'IsoAbbr', 'ISSN (Print)', 'ISSN (Online)'], axis=1, inplace=True)
    return pm


def download_and_compare_pubmed_reference():
    prev_md5 = None
    if os.path.exists(paths.PM_MEDLINE_PATH):
        prev_md5 = get_md5(paths.PM_MEDLINE_PATH)
    _download_pubmed_reference()
    new_md5 = get_md5(paths.PM_MEDLINE_PATH)
    return prev_md5 == new_md5


def _download_pubmed_reference():
    """Download new Pubmed reference (J_Medline.txt) via ftp to data dir."""
    medline_url = "ftp://ftp.ncbi.nih.gov/pubmed/J_Medline.txt"
    with contextlib.closing(urllib_request.urlopen(medline_url)) as r:
        with open(paths.PM_MEDLINE_PATH, 'wb') as f:
            shutil.copyfileobj(r, f)
    _logger.info(f"Updated pubmed reference file from NCBI FTP ('{paths.PM_MEDLINE_PATH}').")


def _write_metadata_file(meta: pd.DataFrame):
    meta.to_csv(paths.PM_META_PATH, sep='\t', compression='gzip', index=True,
                line_terminator='\n')
    _logger.info("Pubmed metadata written to '%s'", paths.PM_META_PATH)


def _load_metadata_file() -> Union[pd.DataFrame, None]:
    if not os.path.exists(paths.PM_META_PATH):
        _logger.info("No metadata path available.")
        return None
    col_type_dict = {'nlmid': str,
                     'uid': str,
                     'main_title': str,
                     'alt_titles_str': str,
                     'in_medline': bool,
                     'is_active': bool,
                     }
    meta = pd.read_csv(paths.PM_META_PATH, sep='\t', compression='gzip',
                       dtype=col_type_dict, lineterminator='\n')
    meta['alt_titles_str'] = meta['alt_titles_str'].fillna('')
    meta.set_index('nlmid', inplace=True)
    return meta


def _load_metadata_for_nlmids(nlmids: Iterable, drop_extra: bool = False,
                              api_key: Union[str, None] = None):
    """Load metadata from file and update via entrez api where necessary.

    Args:
        nlmids: journal NLM IDs.
        drop_extra: remove non-matching nlmids in metadata file.
        api_key: NCBI Entrez API Key (optional)
    """
    meta_prev = _load_metadata_file()
    meta_prev = meta_prev if meta_prev is not None else pd.DataFrame()
    nlmids_prev = set(meta_prev.index)
    nlmids_needed = [i for i in nlmids if i not in nlmids_prev]
    meta_changed = False
    if nlmids_needed:
        findme = pd.DataFrame(data={'nlmid': nlmids_needed})
        _fill_uids(findme, api_key=api_key)
        meta_ext_new = _build_meta_from_uids(findme['uid'], api_key=api_key)
        meta_ext_new.index = findme['nlmid']
        meta_new = _trim_metadata(meta_ext_new)
        if meta_new is None:
            _logger.info(f"No new metadata to add.")
            meta = meta_prev
        else:
            _logger.info(f"Found metadata for {len(meta_new)} new journals.")
            meta = pd.concat([meta_prev, meta_new], axis=0)
            meta_changed = True
    else:
        meta = meta_prev
    if drop_extra:
        extra_nlmids = set(meta.index).difference(nlmids)
        if extra_nlmids:
            _logger.info(f"Dropping metadata for {len(extra_nlmids)} expired records.")
            meta = meta.reindex(nlmids)
            meta_changed = True
    if meta_changed:
        _write_metadata_file(meta)
    meta = meta.reindex(nlmids)
    return meta, meta_changed


def _trim_metadata(meta_ext: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """Extract essential metadata columns for storage/usage."""
    if 'error' in meta_ext.columns:
        valid_rows = meta_ext.error.isnull()
        if not valid_rows.any():
            return None
        # meta = meta_ext[valid_rows].copy()
    # else:
    #     meta = meta_ext.copy()
    meta = meta_ext.copy()
    if len(meta) == 0:
        return None
    main_titles = meta['titlemainlist'].apply(
        lambda v: v[0]['title'].rstrip('.') if pd.notnull(v) else np.nan)
    meta['main_title'] = main_titles
    alt_titles = meta.apply(lambda r: _get_alt_titles(r['titleotherlist'],
                                                      r['main_title']), axis=1)
    meta['alt_titles'] = alt_titles
    meta['alt_titles_str'] = alt_titles.apply(lambda v: '|'.join(v))
    meta['in_medline'] = meta['currentindexingstatus'].map(
        {'Y': True, 'N': False, np.nan: False})
    meta['is_active'] = (meta.endyear == '9999')
    meta_keep_cols = ['uid', 'main_title', 'alt_titles_str', 'in_medline', 'is_active']
    return meta[meta_keep_cols]


def _get_alt_titles(titleotherlist, main_title) -> set:
    if type(titleotherlist) is not list:
        return set()
    alt_set = set()
    for i in titleotherlist:
        alt_title = i['titlealternate']
        if alt_title != main_title:
            alt_set.add(alt_title)
    return alt_set


def _fill_uids(df, api_key=None):  # , save_map=True
    """Use NLM IDs to populate UID column, calling NCBI API if necessary."""
    # uid_dict = _load_uid_dict_from_file()
    df['uid'] = _get_uids_from_numeric_nlmids(df['nlmid'])
    nlmids_unknown = list(df.loc[df.uid.isnull().loc[lambda v: v].index, 'nlmid'])
    if nlmids_unknown:
        _logger.info(f"Looking up {len(nlmids_unknown)} alphanumeric NLM IDs.")
        new_uid_dict = _request_uids_from_nlmids(nlmids_unknown, api_key=api_key)
        _logger.info(f"NLM IDs lookup complete.")
        # uid_dict.update(new_uid_dict)
        # if save_map:
        #     _logger.debug("Updating NLMID > UID dictionary.")
        #     _save_uid_dict_to_file(uid_dict)
        for nlmid in nlmids_unknown:
            if nlmid in new_uid_dict:
                df.loc[df.nlmid == nlmid, 'uid'] = new_uid_dict[nlmid]


def _get_uids_from_numeric_nlmids(nlmid_series: pd.Series) -> pd.Series:
    """Get placeholder uid series from numeric NLM ID series."""
    # if uid_dict is None:
    #     uid_dict = _load_uid_dict_from_file()
    # uids = nlmid_series.map(uid_dict)  # null for those not in dictionary
    uids = pd.Series(data=np.nan, index=nlmid_series.index)
    is_numeric = nlmid_series.str.isnumeric()
    zero_start = nlmid_series.str.startswith('0')
    uids = nlmid_series.where(is_numeric & ~zero_start, uids)
    uids = nlmid_series.str.lstrip('0').where(zero_start & is_numeric, uids)
    return uids


def _build_meta_from_uids(uids, batch_size=400, api_key=None):
    """Build *extended* metadata table for NLM UIDs from NCBI esummary API.

    Includes many metadata columns that will not be used/stored.

    Returns:
        meta (pd.DataFrame): table of journal/book extended metadata.
    """
    # iterate uids in batches
    n_batches = int(np.ceil(len(uids) / batch_size))
    ind = 0
    all_records = []
    for ind_g, g in enumerate(grouper(uids, batch_size)):
        ind += 1
        uids_batch = [i for i in g if i]
        try:
            records = _get_meta_records_from_ids(uids_batch, api_key=api_key)
        except HTTPError414 as err:
            _logger.info(f"414 at index {ind} ({err}), splitting into two requests.")
            n_half = round(len(uids_batch) / 2)
            uids1 = uids_batch[:n_half]
            uids2 = uids_batch[n_half:]
            records1 = _get_meta_records_from_ids(uids1, api_key=api_key)
            records2 = _get_meta_records_from_ids(uids2, api_key=api_key)
            records = records1 + records2
            ind += 1
        _logger.info(f"Progress: finished lookup for batch {ind_g + 1} of {n_batches}")
        all_records.extend(records)
    meta = pd.DataFrame.from_records(all_records)
    return meta


def _get_meta_records_from_ids(uids_batch, api_key=None):
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
    api_kw = {'api_key': api_key} if api_key else {}
    if not api_key:
        _logger.info("API_KEY env variable not present. Attempting Entrez "
                     "request without key.")
    res = http.post(URL_ESUMMARY,
                        params=dict(db='nlmcatalog', id=ids, retmode='json',
                                    **api_kw))
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
            field = 'nlmid' if field == 'NlmId' else field
            val = ''.join(vals[1:]).strip()
            record[field] = val
        yield record


def _request_uids_from_nlmids(nlmids, api_key=None):
    """Use eSearch API to get NLM UIDs from NLM IDs.

    Args:
        nlmids (iterable): NLM IDs as strings.
        api_key (str): NCBI Entrez API KEY (optional)

    Returns:
        dictionary of NLM ID: NLM UID
    """
    uid_dict = dict()
    failed_nlmids = []
    for ind, nlmid in enumerate(nlmids):
        uid = _request_uid_for_single_nlmid(nlmid, api_key=api_key)
        if uid is not None:
            uid_dict[nlmid] = uid
        else:
            failed_nlmids.append(nlmid)
        if not ind % 100:
            _logger.info(f"Download progress: finished UID lookup for index {ind}.")
    if failed_nlmids:
        failed_path = os.path.join(paths.PUBMED_DIR, 'failed_nlmids.txt')
        with open(failed_path, 'w') as out:
            out.write('\n'.join(failed_nlmids))
    return uid_dict


def _request_uid_for_single_nlmid(nlmid: str,
                                  api_key: Union[str, None] = None) -> Union[str, None]:
    """Use eSearch API to get NCBI Entrez UID from NLM ID."""
    from requests.packages.urllib3.util.retry import Retry

    res = http.get(URL_ESEARCH, params=dict(
        db='nlmcatalog', term=nlmid, field='nlmid', api_key=api_key))
    out = xmltodict.parse(res.content)['eSearchResult']
    if 'ERROR' in out:
        _logger.error(f'Failed lookup for {nlmid=} result: {out}. Status={res.status_code}')
        time.sleep(2)  # TRY ONCE MORE
        res = http.get(URL_ESEARCH, params=dict(
            db='nlmcatalog', term=nlmid, field='nlmid', api_key=api_key))
        out = xmltodict.parse(res.content)['eSearchResult']
        if 'ERROR' in out:
            _logger.error(f'Failed 2nd lookup for {nlmid=} result: {out}. Status={res.status_code}')
    if out['Count'] != '1':
        _logger.error(f'Non-single result for {nlmid}.')
        return None
    try:
        uid = out['IdList']['Id']
    except (KeyError, TypeError):
        _logger.error(f'Unexpected NLM ID lookup for {nlmid=} result: {out}. '
                      f'Status: {res.status_code}')
        return None
    return uid
