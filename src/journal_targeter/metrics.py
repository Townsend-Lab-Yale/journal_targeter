import os
import glob
import logging

import pandas as pd

from .paths import METRICS_DIR


_logger = logging.getLogger(__name__)


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
        _logger.info(f"Adding metrics: {metric_cols}.")
        assert ref.index.is_unique
        col_overlap = set(pm.columns).intersection(metric_cols)
        pm.drop(col_overlap, axis=1, inplace=True)
        pm = pm.join(ref[metric_cols], how='left')
    return pm
