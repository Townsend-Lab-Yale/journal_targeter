"""Data consolidation."""
import re
import time
import logging
from functools import partial
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

from . import metrics
from .pubmed import TM
from .ref_loading import identify_user_references
from .lookup import lookup_jane
from .helpers import get_issn_comb

_logger = logging.getLogger(__name__)


def run_queries(query_title=None, query_abstract=None, ris_path=None, refs_df=None):
    """Process queries through Jane and build journal, article and ref tables.

    Args:
        query_title (str): Title text for first Jane search.
        query_abstract (str): Abstract text for second Jane search.
        ris_path (str or stream): RIS path or stream from form data.
        refs_df (pd.DataFrame): (optional) processed ris_path data, prevents
            ris_path processing (ris_path ignored).

    Returns:
        Journals and article results, user citations (j, a, jf, af, refs_df).
            jf: aggregated journal table, combining dups across searches.
            af: aggregated articles table, combining dups across searches.
            refs_df: table of de-duplicated articles from citations file.
    """
    if query_title is None or query_abstract is None or (ris_path is None and
                                                         refs_df is None):
        raise ValueError("Provide query abstract, title and reference info.")
    else:
        refs_kw = dict(ris_path=ris_path) if refs_df is None else \
            dict(refs_df=refs_df)
        journals_t, articles_t, refs_df = process_inputs(
            input_text=query_title, **refs_kw)
        journals_a, articles_a, _ = process_inputs(
            input_text=query_abstract, refs_df=refs_df)

    jf, af = aggregate_jane_journals_articles(journals_t, journals_a,
                                              articles_t, articles_a)

    # FINALIZE MASTER JOURNALS TABLE
    # Add JID to citations table (match on uid, then name<->jane_name, then create new)
    jid_from_uid_dict = {i[1]: i[0] for i in jf['uid'].items() if i[1] != tuple()}
    jid_from_name_dict = {name: uid for uid, name in jf.loc[jf['uid'] == tuple(), 'jane_name'].items()}
    jid_matches = refs_df['uid'].map(jid_from_uid_dict)
    jid_matches = jid_matches.where(~jid_matches.isnull(), refs_df['user_journal'].map(jid_from_name_dict))
    extra_journals = set(refs_df.loc[jid_matches.isnull(), 'user_journal'])
    jid_from_extra_dict = {name: f"c{ind}" for ind, name in enumerate(extra_journals)}
    jid_matches = jid_matches.where(~jid_matches.isnull(), refs_df['user_journal'].map(jid_from_extra_dict))
    refs_df['jid'] = jid_matches

    # COMBINE CITED JOURNALS WITH JANE JOURNALS (-> JFM)
    citations = refs_df[['jid', 'user_journal', 'uid']].value_counts() \
        .reset_index(level=[1, 2]).rename(
            columns={0: 'cited', 'user_journal': 'refs_name'})
    jfm = jf.drop(['single_match'], axis=1).join(
        citations, how='outer', lsuffix='_jane', rsuffix='_refs')
    jfm.insert(0, 'uid', jfm.uid_refs.where(jfm.uid_jane.isnull(), jfm.uid_jane))
    jfm['cited'] = jfm['cited'].fillna(0).astype(int)

    # Add metrics and main title
    jfm = _add_metrics_and_main_title(jfm)

    # ADD CAT
    zero_fill_cols = ['title_only', 'abstract_only', 'title', 'abstract', 'both']
    # jfm.fillna({i: 0 for i in zero_fill_cols})
    for col in zero_fill_cols:
        jfm[col] = jfm[col].fillna(0).astype(int)
    jfm['CAT'] = jfm['cited'] + jfm['abstract'] + jfm['title']
    # ADD PROSPECT
    for impact_col in metrics.METRIC_NAMES:
        jfm[f'p_{impact_col}'] = jfm['CAT'] / (jfm['CAT'] + jfm[impact_col])

    jfm = jfm.sort_values(['CAT', 'sim_sum'], ascending=False).reset_index()  # type: pd.DataFrame

    # Add journal_name, using shortest name option
    jfm.insert(1, 'journal_name', jfm.apply(lambda r: _pick_short_journal_name(
        [r['jane_name'], r['refs_name'], r['main_title']]), axis=1))

    # Add short abbreviation for journals
    abbrv = jfm['uid'].map(TM.pm['abbr'])
    jfm['abbr'] = abbrv.where(~abbrv.isnull(), jfm['journal_name'])
    jfm['abbr'] = jfm['abbr'].apply(lambda v: _get_short_str(v))

    # Fill is_open and in_medline from pm (partial via Jane)
    is_open = jfm['uid'].map(TM.pm['is_open'])
    is_open = is_open.where(jfm['is_oa'].isnull(), jfm['is_oa'])
    jfm['is_open'] = is_open
    jfm.drop(columns=['is_oa'], inplace=True)
    in_medline = jfm['uid'].map(TM.pm['in_medline'])
    in_medline = in_medline.where(jfm['in_medline'].isnull(), jfm['in_medline'])
    jfm['in_medline'] = in_medline

    af['abbr'] = af['jid'].map(jfm['abbr'])

    return jfm, af, refs_df


def aggregate_jane_journals_articles(journals_t, journals_a, articles_t,
                                     articles_a, from_api=True):
    """Build jf, af tables from tall-form journals and articles tables.

    Returns:
        jf (one row per journal), af (one row per article).
    """
    # JF: AGGREGATE JOURNAL TITLE+ABSTRACT RESULTS
    j, a = _concat_jane_data(journals_t, journals_a, articles_t, articles_a)

    temp = j.copy()
    temp['conf_sum'] = temp['confidence']  # placeholder for conf sum aggregation
    groupby_col = 'jid'
    get_first = ['jane_name', 'influence', 'tags', 'uid', 'single_match', 'is_oa']
    # get_first += list(metrics.METRIC_NAMES)
    if from_api:
        get_first.extend(['in_medline', 'in_pmc'])
    get_sum = ['sim_sum', 'conf_sum']
    get_max = ['sim_max']
    get_min = ['sim_min']
    get_tuple = ['confidence', 'sims', 'pc_lower']  # inc categ
    # Basic aggregation columns
    fn_dict = {i: _get_first_series_val for i in get_first}
    fn_dict.update({i: pd.Series.sum for i in get_sum})
    fn_dict.update({i: pd.Series.max for i in get_max})
    fn_dict.update({i: pd.Series.min for i in get_min})
    agg_cols = temp.groupby(groupby_col).agg(fn_dict)
    # Tuple aggregation
    records = []
    for ind, g in temp.groupby(groupby_col):
        rec = _get_agg_record_for_field_list(g, groupby_col, ind, get_tuple)
        records.append(rec)
    tuple_cols = pd.DataFrame.from_records(records, index=groupby_col)
    jf = pd.concat([agg_cols, tuple_cols], axis=1)
    jf['conf_pc'] = jf['conf_sum'] / jf['conf_sum'].sum() * 100
    if not from_api:
        jf['tags'] = jf['tags'].fillna('')
        jf['in_medline'] = jf['tags'].map(lambda v: True if 'Medline' in v else False)
        jf['in_pmc'] = jf['tags'].map(lambda v: True if 'PMC' in v else False)
    jf['conf_title'] = jf['confidence'].map(partial(_get_categ_confidence, categ='title'))
    jf['conf_abstract'] = jf['confidence'].map(partial(_get_categ_confidence, categ='abstract'))

    af = pd.DataFrame.from_records(a.groupby('a_id').apply(_get_article_record).values)

    # Add jane article counts by subgroup to journals table
    bool_cols = ['t_only', 'a_only', 'in_title', 'in_abstract', 'in_both']
    n_articles = af.groupby('jid')[bool_cols].sum() \
        .astype(int).rename(columns={'in_title': 'title',
                                     't_only': 'title_only',
                                     'in_abstract': 'abstract',
                                     'a_only': 'abstract_only',
                                     'in_both': 'both'})
    jf = jf.join(n_articles, on='jid', how='left')

    af['PMID'] = af['a_id'].str[5:]
    af['authors_short'] = af['authors'].apply(_get_pm_authors_short)
    return jf, af


def process_inputs(input_text=None, ris_path=None, refs_df=None):
    """Get similar journals and perform matching (user<>pubmed; JANE<>pubmed)."""

    # Process user citations if refs_df not provided
    if refs_df is None:
        refs_df = identify_user_references(ris_path)

    # Get matches and scores
    journals, articles = lookup_jane(input_text)

    # Add pubmed matching info
    _logger.info("Matching JANE titles to PubMed titles.")
    match_jane = TM.match_titles(journals.jane_name)
    journals = pd.concat([journals, match_jane], axis=1)

    return journals, articles, refs_df


def _pick_short_journal_name(name_options):
    """Get shortest string among options, ignoring null values.
    >>> _pick_short_journal_name( \
        ['Clinical Infectious Diseases', 'Clin Infect Dis', \
         'Clinical infectious diseases : an official publication of the Infectious Diseases Society of America'])
    'Clin Infect Dis'
    """
    name_lengths = [(name, len(name)) for name in name_options if not pd.isnull(name)]
    name_lengths.sort(key=lambda i: i[1], reverse=False)
    return name_lengths[0][0]


def _concat_jane_data(journals_t, journals_a, articles_t, articles_a):
    """Merge title and abstract tables into narrow journal and article tables."""
    j1 = journals_t.reset_index()
    j1.insert(0, 'source', 'title')
    j2 = journals_a.reset_index()
    j2.insert(0, 'source', 'abstract')
    j = pd.concat([j1, j2], axis=0)

    # ADD 'jid' to capture unique journals (by jane_name, influence score)
    j_tuples = [tuple(i) for i in j[['jane_name', 'influence']].fillna(-1)
                .drop_duplicates().values]
    jid_map = {i: str(ind) for ind, i in enumerate(j_tuples)}
    jids = [jid_map[tuple(i)] for i in j[['jane_name', 'influence']]
            .fillna(-1).values]
    j.insert(0, 'jid', jids)
    source_rank_dict = j.set_index(['source', 'j_rank'])['jid'].to_dict()

    # Get combined ARTICLES table
    a1 = articles_t.reset_index()
    a1.insert(0, 'source', 'title')
    a2 = articles_a.reset_index()
    a2.insert(0, 'source', 'abstract')
    a = pd.concat([a1, a2], axis=0)
    # Add jid to articles table
    a_jids = [source_rank_dict[tuple(i)] for i in a[['source', 'j_rank']].values]
    a.insert(0, 'jid', a_jids)
    return j, a


def _add_metrics_and_main_title(df):
    use_pm_cols = ['main_title'] + [i for i in metrics.METRIC_NAMES if i != 'influence']
    out = df.join(TM.pm[use_pm_cols], how='left', on='uid')
    return out


def _get_categ_confidence(conf_str, categ):
    """Get confidence value from aggregated confidence string.

    Examples:
        >>> _get_categ_confidence('title:2 abstract:12', 'abstract')
        '12'
        >>> _get_categ_confidence('abstract:12', 'title')
        ''
    """
    match = re.search(fr'{categ}:(\d+)', conf_str)
    return match.groups()[0] if match else ''


def _get_article_record(g):
    od = OrderedDict()
    od.update(g[['jid', 'title', 'authors', 'year', 'a_id', 'url']]
              .apply(lambda s: s.iloc[0]).to_dict())
    od['sim_max'] = g['sim'].max()
    in_title, in_abstract = [i in g['source'].values for i in ['title', 'abstract']]
    categ = 'both' if in_title and in_abstract else 'title_only' if in_title else 'abstract_only'

    od.update({'in_title': in_title,
               'in_abstract': in_abstract,
               'in_both': in_title and in_abstract,
               't_only': in_title and not in_abstract,
               'a_only': in_abstract and not in_title,
               'categ': categ,
               })
    return od


def _get_initials(v):
    first_chars = [i[0] for i in v.split(' ')]
    initials = ''.join([i for i in first_chars if i.isalpha()])
    return initials


def _get_article_categ(r):
    if r.in_both:
        return 'both'
    if r.in_title:
        return 'title'
    if r.in_abstract:
        return 'abstract'
    return '???'


def _get_pm_authors_short(author_str):
    author_list = author_str.split(',')
    n_authors = len(author_list)
    if n_authors == 1:
        return author_str
    surnames = [''.join(i.strip().split(' ')[:-1]) for i in author_list]
    if n_authors == 2:
        return ' & '.join(surnames)
    return f"{surnames[0]} et al"


def _get_short_str(v, maxlen=25):
    if len(v) > 25:
        return v[:maxlen - 1] + '…'
    return v


def _get_first_series_val(s):
    return s.iloc[0]


def _get_agg_str_for_field(group_df, field_name=None):
    #     return tuple(g[['categ', field]].values.flatten())
    use_rows = ~group_df[field_name].isnull()
    mod_field_series = group_df.loc[use_rows, field_name].convert_dtypes()
    str_series = group_df[use_rows].source + ':' + mod_field_series.astype(str)
    return ' '.join(str_series)


def _get_agg_record_for_field_list(group_df, index_name, index_value, field_list):
    out = {index_name: index_value}
    for field in field_list:
        out[field] = _get_agg_str_for_field(group_df, field)
    return out


def _coerce_issn_to_numeric_string(issn):
    """
    >>> _coerce_issn_to_numeric_string('123-45678')
    '12345678'
    >>> _coerce_issn_to_numeric_string('004-4586X')
    '0044586X'
    >>> _coerce_issn_to_numeric_string('***-*****')
    ''
    """
    if pd.isnull(issn):
        return np.nan
    assert(type(issn) is str), "ISSN must be a string."
    issn = ''.join([i for i in issn if i.isnumeric() or i in {'X', 'x'}])
    return issn.upper()


def _get_query_table(titles=None, issn_print=None, issn_online=None):
    """

    Args:
        titles:
        issn_print:
        issn_online:

    Returns:

    """
    # BUILD QUERY DATAFRAME
    if titles is None or issn_print is None:
        raise ValueError("titles and issn_print are required.")
    issn_print = [_coerce_issn_to_numeric_string(i) for i in issn_print]
    data = {'title': list(titles), 'issn_print': issn_print}
    has_issn_online = issn_online is not None
    if has_issn_online:
        issn_online = [_coerce_issn_to_numeric_string(i) for i in issn_online]
        data.update({'issn_online': issn_online})
        issn_comb = get_issn_comb(pd.Series(issn_print), pd.Series(issn_online))
        data.update({'issn_comb': issn_comb})
    df = pd.DataFrame(data)
    return df


def lookup_uids_from_title_issn(titles=None,
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

    match = TM.match_titles(list(titles), n_processes=n_processes)
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
    pm = TM.pm
    pm_issnp1 = pm.issn_print.dropna().drop_duplicates(keep=False).reset_index()  # print issn uniquely points to uid
    pm_issno1 = pm.issn_online.dropna().drop_duplicates(keep=False).reset_index()  # online issn uniquely points to uid

    # ISSN MATCHING
    df = _get_query_table(titles=titles, issn_print=issn_print, issn_online=issn_online)
    has_comb_issn = 'issn_comb' in df.columns
    if has_comb_issn:
        # issn_comb matching useful when issnp repeated but issnp+issno seen once
        pm_issn_comb1 = pm['issn_comb'].drop_duplicates(keep=False).reset_index()
        # skip repeated issn_comb pairs in query table
        df_issn_comb = df['issn_comb'].drop_duplicates(keep=False).reset_index()
        issnc = pm_issn_comb1.merge(df_issn_comb, how='inner', on='issn_comb') \
            .set_index('index')['uid']
        match['on_issnc'] = issnc
        # Add Print ISSN matching info
        df_issnp = df['issn_print'].dropna().drop_duplicates(keep=False).reset_index()
        issnp = pm_issnp1.merge(df_issnp, how='inner', on='issn_print').set_index('index')['uid']
        match['on_issnp'] = issnp
        # Add Online ISSN matching info
        df_issno = df['issn_online'].dropna().drop_duplicates(keep=False).reset_index()
        issno = pm_issno1.merge(df_issno, how='inner', on='issn_online').set_index('index')['uid']
        match['on_issno'] = issno
    else:  # single issn provided per title
        issns = pd.DataFrame({
            'uid': list(pm_issnp1.uid) + list(pm_issno1.uid),
            'issn': list(pm_issnp1.issn_print) + list(pm_issno1.issn_online),
            'categ': list(np.repeat('print', len(pm_issnp1))) + list(np.repeat('online', len(pm_issno1))),
        })
        # all issn-uid pairs
        issns = issns.groupby(['uid', 'issn']).aggregate(lambda s: ','.join(set(s))).reset_index()
        # remove issns that are in more than one pair
        ambig_issns = issns.issn.value_counts().loc[lambda v: v > 1].index
        issns1 = issns[~issns['issn'].isin(ambig_issns)]
        issn_dict = issns1.set_index('issn')['uid'].to_dict()
        match['on_issnp'] = df['issn_print'].map(issn_dict)

    # Add boolean columns for ISSN variant matches
    if has_comb_issn:
        match['bool_issnc'] = ~match.on_issnc.isnull()
        match['bool_issno'] = ~match.on_issno.isnull()
    match['bool_issnp'] = ~match.on_issnp.isnull()

    # Count matches and categorize discrepancies for each scopus ID
    var_cols = ('issnc', 'issnp', 'issno') if has_comb_issn else ('issnp',)
    categs = pd.DataFrame.from_records(
        match.apply(lambda r: _classify_ids(r, cols=var_cols), axis=1).values,
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

    # Handle cases where multiple scopus IDs map to single NLM UID.
    ambig_uids = set((match['uid'].value_counts() > 1).loc[lambda v: v].index.values)
    if ambig_uids and resolve_uid_conflicts:
        conflict_uids = set((match['uid'].value_counts() > 1).loc[lambda v: v].index.values)
        conflicts = match[match['uid'].isin(conflict_uids)].reset_index()
        conflicts['score'] = conflicts.apply(_get_match_score, axis=1)
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
        assert match.uid.value_counts().max() == 1, "Scopus matching still includes conflicts."
    elif ambig_uids and not resolve_uid_conflicts:
        _logger.info(f"Note: {len(ambig_uids)} records map to same UID.")
    elif not ambig_uids:
        _logger.info("No conflicting matches found when mapping to pubmed.")
    time_end = time.perf_counter()
    _logger.info(f"Matching to Pubmed UIDs took {time_end - time_start:.1f} seconds")
    n_unmatched = match.uid.isnull().sum()
    n_matched = len(match) - n_unmatched
    _logger.info(f"Successfully matched {n_matched} journals, leaving {n_unmatched} "
                 f"not linked to pubmed.")
    return match


def _get_match_score(r):
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




def _classify_ids(r, cols=('issnc', 'issnp', 'issno')):
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
    categ = get_categ_from_uid_dict(d)
    return n_vals, categ


def get_categ_from_uid_dict(d):
    dd = defaultdict(list)
    for i in d:
        dd[d[i]].append(i)
    categ = '|'.join(sorted(['_'.join(sorted(i)) for i in dd.values()]))
    return categ
