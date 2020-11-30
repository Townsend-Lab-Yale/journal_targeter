import os
import glob
import logging
from collections import OrderedDict

import pandas as pd

from . import METRICS_DIR


_logger = logging.getLogger(__name__)

METRIC_NAMES = OrderedDict()  # auto-populated. column name -> display name


def save_metric_map(uid_dict=None, metric_df=None, metric_col_map=None,
                    metric_basename=None):
    """Create metric table with `uid`, `source_id`, `<metric1>`, `<metric2>`, ...

    Args:
        uid_dict (dict): dictionary of NLM UID -> metric source ID
        metric_df (pd.DataFrame): table with metric info, e.g. Scopus data
        metric_col_map (dict): metric column -> output name, for desired columns
            in metric_df.
        metric_basename: output reference file basename (<basename>.tsv)

    Returns:
        Reference table (pd.DataFrame) as saved to file.
    """
    ref = pd.Series(uid_dict)
    ref.index.name = 'uid'
    ref.name = 'source_id'
    ref = pd.DataFrame(ref)
    ref = ref.join(metric_df[list(metric_col_map)], on='source_id', how='left')
    ref.rename(columns=metric_col_map, inplace=True)
    # SAVE TO TSV
    if not os.path.exists(METRICS_DIR):
        os.makedirs(METRICS_DIR, exist_ok=True)
    outpath = os.path.join(METRICS_DIR, f"{metric_basename}.tsv")
    ref.reset_index().to_csv(outpath, sep='\t', index=False)


def update_metrics_scopus():
    from .scopus import SCOP
    from .pubmed import load_scopus_map
    scop_dict = load_scopus_map()
    save_metric_map(uid_dict=scop_dict, metric_df=SCOP,
                    metric_col_map={'citescore': 'citescore'},
                    metric_basename='scopus')


def update_metrics_jcr(uid_jcr_dict=None, jcr_df=None):
    save_metric_map(uid_dict=uid_jcr_dict, metric_df=jcr_df,
                    metric_col_map={'journalImpactFactor': 'impact'},
                    metric_basename='jcr')


def add_metrics_to_pm(pm):
    """Add all metric data in metrics folder to NLM/pubmed reference table."""
    metrics_paths = glob.glob(os.path.join(METRICS_DIR, '*.tsv'))
    for path in metrics_paths:
        ref = pd.read_csv(path, sep='\t', dtype={'uid': str})
        ref.set_index('uid', inplace=True)
        metric_cols = [i for i in ref.columns if i != 'source_id']
        _logger.info(f"Adding metrics: {metric_cols}")
        assert ref.index.is_unique
        col_overlap = set(pm.columns).intersection(metric_cols)
        pm.drop(col_overlap, axis=1, inplace=True)
        pm = pm.join(ref[metric_cols], how='left')
    return pm


def update_metric_list():
    """Read reference metric files and identify metric columns."""
    global METRIC_NAMES
    metrics_paths = glob.glob(os.path.join(METRICS_DIR, '*.tsv'))
    metric_list = []
    for path in metrics_paths:
        with open(path, 'r') as infile:
            line = infile.readline()
            metrics = line.strip().split('\t')[2:]
            metric_list.extend(metrics)
    METRIC_NAMES.clear()
    METRIC_NAMES.update({i: i.title() for i in metric_list})
    if 'citescore' in METRIC_NAMES:
        METRIC_NAMES['citescore'] = 'CiteScore'
    METRIC_NAMES['influence'] = 'Influence'


update_metric_list()