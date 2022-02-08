import os
import glob
import logging
from datetime import datetime
from typing import Union, List

import pandas as pd

from .paths import METRICS_DIR
from .sherpa_romeo import load_sherpa_id_map
from.doaj import load_doaj_table

_logger = logging.getLogger(__name__)


class ColumnMissingError(Exception):
    pass


class RefTable:
    """Holds journal metadata from database such as Scopus or JCR.

    Columns are stripped down specified matching and metadata columns."""
    def __init__(self,
                 source_name=None,
                 df=None,
                 title_col=None,
                 col_metrics=None,
                 col_other=None,
                 issn_col=None,
                 issn_print=None,
                 issn_online=None,
                 index_is_uid=False,
                 rename_dict=None,
                 version_str=None,
                 ):
        """Populate RefTable with data and column specifications.

        Args:
            source_name (str): e.g. scopus, jcr
            df (pd.DataFrame): Input data table, including (but not limited to)
                columns specified in arguments.
            title_col (str): Journal title column
            col_metrics (list): Column names with metrics to keep.
            col_other (list): OPTIONAL Additional non-metric columns to keep.
            issn_col (str): OPTIONAL Generic ISSN. Only provide when you don't
                know if ISSN is print or online version.
            issn_print (str): OPTIONAL Print ISSN column name
            issn_online (str): OPTIONAL E-ISSN column name
            index_is_uid (bool): Treat index as source-specific UID.
                If False, records will be assumed to have no UID.
            rename_dict (dict): OPTIONAL source columns -> new columns.
                Renaming is performed first, so column arguments should specify
                post-renaming column names.
            version_str (str): OPTIONAL Identifier for this source version, e.g. date string.
        """
        provided = [('source_name', bool(source_name)),
                    ('title_col', title_col is not None),
                    ('col_metrics', bool(col_metrics)),
                    ('df', type(df) is pd.DataFrame and len(df)),
                    ]
        missing_args = [i[0] for i in provided if not i[1]]
        if missing_args:
            raise ValueError(f"Missing/empty arguments: {missing_args}.")

        self.source_name = source_name
        self.rename_for_merge_dict = rename_dict or dict()
        self.title_col = title_col
        self.issn_col = issn_col
        self.issn_print = issn_print
        self.issn_online = issn_online
        self.index_is_uid = index_is_uid
        if type(col_metrics) is str:
            col_metrics = [col_metrics]  # force list type
        self.col_metrics = col_metrics
        if type(col_other) is str:
            col_other = [col_other]  # force list type
        self.col_other = col_other if col_other is not None else []
        self.version_str = version_str or self._create_version_str()
        self.df = self._get_trimmed_df(df)

    def _get_trimmed_df(self, df_full):
        """Reduce table to essential metadata and matching columns."""
        if self.rename_for_merge_dict:
            df_full = df_full.rename(columns=self.rename_for_merge_dict)
        keep_cols = ([self.title_col] + self.col_metrics + self._issn_cols
                     + self.col_other)
        missing_cols = set(keep_cols).difference(df_full.columns)
        if missing_cols:
            raise ColumnMissingError(f"{missing_cols} not found in table.")
        drop_cols = [i for i in df_full.columns if i not in keep_cols]
        df = df_full[keep_cols] if drop_cols else df_full
        _logger.info(f"{len(df)} rows loaded into {self.source_name} table.")
        if not self.index_is_uid and df.index.name is None:
            df.index.set_names(f'{self.source_name}_ix')
        # Ensure string-type index if index_is_uid
        if self.index_is_uid and type(df.index[0]) != str:
            df = df.copy()
            df.index = df.index.astype(str)
        return df

    @property
    def _issn_cols(self) -> List:
        temp_cols = [self.issn_col, self.issn_print, self.issn_online]
        return [i for i in temp_cols if i]

    @staticmethod
    def _create_version_str():
        return datetime.strftime(datetime.utcnow(), '%Y-%m-%d_%H%M')

    def __repr__(self):
        return f"<RefTable {self.source_name} {self.version_str}>"


class TableMatcher:
    """Performs matching and updates saved metadata and source<->NLM uid map.

    Attributes:
        meta_matchable: table of titles and issns as input for matching
        meta_full: table with final metadata columns indexed by source_id
        meta_matched: table with final metadata columns, indexed by NLM uid

        source_matched: table of matched titles/issns
        source_unmatched: table of unmatched titles/issns

        has_map_file (bool): id map file exists
        has_meta_file (bool): metadata file exists
        map_path (str): path for matched ids to be saved as tsv
        meta_path (str): path for matched metadata to be saved as tsv
    """
    def __init__(self, rt: RefTable) -> None:
        """Initialize TableMatcher with source data in RefTable object."""
        self.source_name = rt.source_name
        self.col_metrics = rt.col_metrics
        self.col_other = rt.col_other
        self.index_name = rt.df.index.name
        self.title_col = rt.title_col
        self.has_uid = rt.index_is_uid

        issn_kw = {}
        if rt.issn_col:
            issn_kw.update({'issn_print': rt.issn_col})
        elif rt.issn_print:
            issn_kw = {'issn_print': rt.issn_print}
            if rt.issn_online:
                issn_kw.update({'issn_online': rt.issn_online})
        self.issn_kw = issn_kw

        issn_cols = list(issn_kw.values())
        self.meta_matchable = rt.df[[self.title_col] + issn_cols]
        self.meta_full = rt.df[rt.col_metrics + rt.col_other]

        self.id_map = self.read_map()

    @property
    def meta_matched(self) -> Union[pd.DataFrame, None]:
        if self.id_map is None:
            return
        return self.meta_full.join(self.id_map, how='inner').set_index('nlmid')

    @property
    def source_matched(self) -> Union[pd.DataFrame, None]:
        if self.id_map is None:
            return
        matched_inds = self.meta_matchable.index.intersection(self.id_map.index)
        return self.meta_matchable.loc[matched_inds]

    @property
    def source_unmatched(self) -> Union[pd.DataFrame, None]:
        if self.id_map is None:
            return self.meta_matchable
        matched = set(self.id_map.index) if self.id_map is not None else set()
        unmatched_inds = self.meta_matchable.index.difference(matched)
        return self.meta_matchable.loc[unmatched_inds]

    def match_missing(self, save: bool = True, n_processes: int = 3) -> pd.Series:
        """Use titles and ISSNs to expand id_map (source ID -> NLM uid).

        Args:
            save: save resulting matched data to files.
            n_processes: Number of processes used for matching.

        Returns:
            Pandas series with {source_id_name} as index, uid as values.
        """
        _logger.info('Start matching for %s data.', self.source_name)
        from .reference import TM
        res = TM.lookup_uids_from_title_issn(
            self.source_unmatched[self.title_col], n_processes=n_processes,
            **{key: self.source_unmatched[val] for
               key, val in self.issn_kw.items()})
        new_matches = res.set_index(self.source_unmatched.index)['uid'].dropna()
        new_matches.name = 'nlmid'
        n_initial = len(self.id_map) if self.id_map is not None else 0
        n_new = len(new_matches)
        _logger.info(f"Adding {n_new} matches to {n_initial} record ID map.")
        self.id_map = pd.concat([self.id_map, new_matches]).drop_duplicates()
        if save:
            if len(new_matches) == 0:
                _logger.info("No new matches found -- skipping save.")
            else:
                self.save_matches()
        return new_matches

    def save_matches(self):
        """Save ID map (source ID -> NLM ID) and metrics/metadata to file."""
        if self.has_uid:
            self._save_map()
        self._save_meta()

    def _save_map(self) -> None:
        """Save id_map to file."""
        self.id_map.to_csv(self.map_path, sep='\t', line_terminator='\n',
                           encoding='utf8', compression='gzip')
        _logger.info(f"Mapping data for {self.source_name} saved to '{self.map_path}'.")

    def _save_meta(self) -> None:
        """Save metrics/metadata to file, indexed by NLM UID."""
        self.meta_matched.to_csv(self.meta_path, sep='\t', line_terminator='\n',
                                 encoding='utf8', compression='gzip')
        _logger.info(f"Metrics data for {self.source_name} saved to '{self.meta_path}'.")

    @property
    def map_path(self) -> Union[str, None]:
        if not self.has_uid:
            return
        return os.path.join(METRICS_DIR, f"{self.source_name}_map.tsv.gz")

    @property
    def meta_path(self) -> str:
        return os.path.join(METRICS_DIR, f"{self.source_name}_meta.tsv.gz")

    def has_map_file(self) -> bool:
        return os.path.exists(self.map_path)

    def has_meta_file(self) -> bool:
        return os.path.exists(self.meta_path)

    def read_map(self) -> Union[pd.Series, None]:
        """Load previous ID map (source ID -> NLM UID) from file."""
        if not self.has_uid or not self.has_map_file():
            _logger.info("No map file to load.")
            return
        id_map = pd.read_csv(self.map_path, sep='\t', lineterminator='\n',
                             encoding='utf8', compression='gzip', dtype=str)
        assert(len(id_map.columns) == 2), "Two columns required in ID map data."
        other_col = set(id_map.columns).difference({'nlmid'}).pop()
        id_map = id_map.set_index(other_col).sort_index()['nlmid']
        return id_map

    def read_meta(self) -> Union[None, pd.DataFrame]:
        """Not used."""
        if not self.has_meta_file():
            _logger.info("No meta file to load.")
            return

        meta = pd.read_csv(self.meta_path, sep='\t', lineterminator='\n',
                           encoding='utf8', compression='gzip',
                           dtype={'nlmid': str}).set_index('nlmid')
        return meta


class MasterTable:
    """Holds master table of journal uids, titles, metrics, other key metadata.

    Attributes:
        df: master table as DataFrame.
        metric_list: column names for impact metrics in df.
    """

    def __init__(self, pm_full: pd.DataFrame = None):
        self.df = None
        self.metric_list = None
        self.other_meta_list = None
        if pm_full is not None:
            self.init_data(pm_full)

    def init_data(self, pm_full: pd.DataFrame = None):
        master = pm_full[['main_title', 'abbr', 'in_medline']].copy()
        meta_paths = self._fetch_meta_paths(scopus_first=True)
        metric_cols = []
        other_cols = []
        for path in meta_paths:
            meta_tmp = pd.read_csv(path, sep='\t', lineterminator='\n', compression='gzip',
                                   dtype={'nlmid': str}).set_index('nlmid')
            new_metric_cols = list(meta_tmp.select_dtypes('number').columns)
            new_other_cols = [i for i in meta_tmp.columns if i not in new_metric_cols]
            metric_cols.extend(new_metric_cols)
            other_cols.extend(new_other_cols)
            master = master.join(meta_tmp, how='left')
        master['sr_id'] = load_sherpa_id_map()
        doaj = load_doaj_table()
        master = master.join(doaj, how='left')
        master['is_open'] = master['doaj_compliant'] == 'Yes'
        self.df = master
        self.metric_list = metric_cols
        self.other_meta_list = other_cols
        _logger.info(f"Created master table. Metrics: %s; Other: %s.",
                     metric_cols, other_cols)

    def get_uid_pretty(self, uid):
        from .plot import _URL_NLM_BK, _URL_ROMEO
        res = self.df.loc[uid]
        if not len(res):
            return ''
        url_nlm = _URL_NLM_BK.replace('@uid', uid)
        header = pd.Series(data=[uid, url_nlm], index=['nlmid', 'NLM URL'])
        res = pd.concat([header, res]).fillna('')
        url_sr = _URL_ROMEO.replace('@sr_id', res.sr_id) if res.sr_id else ''
        res['sr_id'] = url_sr
        res.rename({
            'sr_id': 'Romeo URL',
            'url_doaj': 'DOAJ URL',
        }, inplace=True)
        s = []
        for name, val in res.iteritems():
            s.append(f'{name: <18} {val}')
        return '\n'.join(s)

    @staticmethod
    def _reduce_pubmed_table(pm):
        keep_cols = ['main_title', 'abbr', 'in_medline']
        return pm[keep_cols]

    @staticmethod
    def _fetch_meta_paths(scopus_first=True) -> list:
        """Get meta paths in metrics directory. Optionally list scopus first."""
        meta_paths = glob.glob(os.path.join(METRICS_DIR, '*_meta.tsv.gz'))
        if scopus_first:
            ind_scopus = 0
            for ind, path in enumerate(meta_paths):
                if 'scopus' in path:
                    ind_scopus = ind
                    break
            if ind_scopus:
                path_scopus = meta_paths.pop(ind_scopus)
                meta_paths.insert(0, path_scopus)
        return meta_paths
