"""Data consolidation."""
import os
import re
import json
import logging
from functools import partial
from collections import defaultdict, OrderedDict

import pandas as pd

from . import scopus, DATA_DIR
from .pubmed import TM, MATCH_JSON_PATH, MATCH_PICKLE_PATH
from .scopus import SCOP
from .ref_loading import identify_user_references
from .lookup import lookup_jane


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
            j: tall-form journal and articles, one row per search-result pair.
            a: tall-form journal and articles, one row per search-result pair.
            jf: aggregated journal table, combining dups across searches.
            af: aggregated articles table, combining dups across searches.
            refs_df: table of de-duplicated articles from citations file.
    """
    if query_title is None or query_abstract is None or ris_path is None:
        _logger.debug("Loading demo data")
        journals_t, articles_t, refs_df = process_inputs(use_title=True)
        journals_a, articles_a, _ = process_inputs(use_title=False, refs_df=refs_df)
    else:
        refs_kw = dict(ris_path=ris_path) if refs_df is not None else \
            dict(refs_df=refs_df)
        journals_t, articles_t, refs_df = process_inputs(
            input_text=query_title, use_title=True, **refs_kw)
        journals_a, articles_a, _ = process_inputs(
            input_text=query_abstract, use_title=False, refs_df=refs_df)

    # Get combined JOURNALS table
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

    # Add short abbreviation for journals
    abbrv = j['uid'].map(TM.pm['IsoAbbr'])
    j['abbr'] = abbrv.where(~abbrv.isnull(), j['journal_name'])
    j['abbr'] = j['abbr'].apply(lambda v: _get_short_str(v))

    # Get combined ARTICLES table
    a1 = articles_t.reset_index()
    a1.insert(0, 'source', 'title')
    a2 = articles_a.reset_index()
    a2.insert(0, 'source', 'abstract')
    a = pd.concat([a1, a2], axis=0)
    # Add jid to articles table
    a_jids = [source_rank_dict[tuple(i)] for i in a[['source', 'j_rank']].values]
    a.insert(0, 'jid', a_jids)

    jf, af = aggregate_journals_articles(j, a)

    return j, a, jf, af, refs_df


def aggregate_journals_articles(j, a):
    """Build jf, af tables from tall-form journals and articles tables.

    Returns:
        jf, af.
    """
    # JF: AGGREGATE JOURNAL TITLE+ABSTRACT RESULTS
    temp = j.copy()
    temp['conf_sum'] = temp['confidence']  # placeholder for conf sum aggregation
    groupby_col = 'jid'
    get_first = ['journal_name', 'citescore', 'influence', 'tags', 'abbr',
                 'cited', 'uid', 'scopus_id', 'single_match', 'is_oa']
    get_sum = ['n_articles', 'sim_sum', 'conf_sum']
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
    n_unique = j.merge(a[['jid', 'a_id']], how='left', on=['jid']) \
        .groupby(groupby_col)['a_id'].nunique()
    jf['n_unique'] = n_unique
    jf['conf_pc'] = jf['conf_sum'] / jf['conf_sum'].sum() * 100
    jf['cites+hits'] = jf['cited'] + jf['n_unique']
    jf['tags'] = jf['tags'].fillna('')
    jf['is_open'] = jf['tags'].map(lambda v: '\u2714' if 'open' in v else '')
    jf['in_medline'] = jf['tags'].map(lambda v: '\u2714' if 'Medline' in v else '')
    jf['in_pmc'] = jf['tags'].map(lambda v: '\u2714' if 'PMC' in v else '')
    jf['conf_title'] = jf['confidence'].map(partial(_get_categ_confidence, categ='title'))
    jf['conf_abstract'] = jf['confidence'].map(partial(_get_categ_confidence, categ='abstract'))
    jf['initials'] = jf.abbr.apply(_get_initials)

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
    jf['CAT'] = jf['cited'] + jf['abstract'] + jf['title']

    jf = jf.sort_values(['CAT', 'conf_pc'], ascending=False).reset_index()

    af['abbr'] = af['jid'].map(jf.set_index('jid')['abbr'])
    af['PMID'] = af['a_id'].str[5:]
    af['authors_short'] = af['authors'].apply(_get_pm_authors_short)
    return jf, af


def process_inputs(input_text=None, ris_path=None, use_title=True, refs_df=None):
    """Get similar journals and perform matching (user<>pubmed; JANE<>pubmed).

    """
    # Get matches and scores
    journals, articles = lookup_jane(input_text)

    # Add pubmed matching info
    _logger.info("Matching JANE titles to PubMed titles.")
    match_jane = TM.match_titles(journals.jane_name)
    journals = pd.concat([journals, match_jane], axis=1)

    # Process user citations if refs_df not provided
    if refs_df is None:
        refs_df = identify_user_references(ris_path)

    # Add citation counts
    n_cites_dict = refs_df.groupby('uid').size().to_dict()
    journals['cited'] = journals['uid'].map(n_cites_dict).fillna(0).astype(int)

    # Add citescore
    journals = journals.join(TM.pm[['main_title', 'scopus_id']], how='left', on='uid') \
        .join(SCOP[['journal_name', 'citescore']], how='left', on='scopus_id')
    # Prioritize scopus name, then jane_name if null
    journals['journal_name'] = journals.journal_name.where(~journals.journal_name.isnull(),
                                                           journals.jane_name)
    return journals, articles, refs_df


def _get_categ_confidence(conf_str, categ):
    """Get confidence value from aggregated confidence string.

    Examples:
        >>> _get_categ_confidence('title:2 abstract:12', 'abstract')
        12
        >>> _get_categ_confidence('abstract:12', 'title')
        ''
    """
    match = re.search(f'{categ}:(\d+)', conf_str)
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


def build_uid_match_table(n_processes=4, write_files=False):
    """Get NLM UIDs for all Scopus journal IDs based on Scopus names and ISSNs.

    Args:
        n_processes (int): number of processes. Use mutiprocessing if >1.
        write_files (bool): write match table to pickle and json in DATA dir.
    """
    pm = TM.pm
    scop = scopus.load_scopus_journals_reduced()
    # Attempt match on Scopus journal titles
    titles = scop.journal_name.values
    match = TM.match_titles(titles, n_processes=n_processes)
    match.rename(columns={
        'uid': 'on_name',
        'categ': 'name_method',
        'single_match': 'on_name_single',
        }, inplace=True)
    # Add COMBINED ISSN matching info
    pm_issn_comb = pm.loc[pm.is_active, 'issn_comb'].drop_duplicates(keep=False).reset_index()
    scop_issn_comb = scop.issn_comb.drop_duplicates(keep=False).reset_index()
    issnc = pm_issn_comb.merge(scop_issn_comb, how='inner', on='issn_comb').set_index('scopus_id')['uid']
    match['on_issnc'] = issnc
    # Add Print ISSN matching info
    pm_issnp = pm.issn_print.dropna().drop_duplicates(keep=False).reset_index()
    scop_issnp = scop.issn_print.dropna().drop_duplicates(keep=False).reset_index()
    issnp = pm_issnp.merge(scop_issnp, how='inner', on='issn_print').set_index('scopus_id')['uid']
    match['on_issnp'] = issnp
    # Add Online ISSN matching info
    pm_issno = pm.issn_online.dropna().drop_duplicates(keep=False).reset_index()
    scop_issno = scop.issn_online.dropna().drop_duplicates(keep=False).reset_index()
    issno = pm_issno.merge(scop_issno, how='inner', on='issn_online').set_index('scopus_id')['uid']
    match['on_issno'] = issno
    # Add boolean for ISSN variant matches
    match['bool_name'] = match.name_method != 'unmatched'
    match['bool_issnc'] = ~match.on_issnc.isnull()
    match['bool_issnp'] = ~match.on_issnp.isnull()
    match['bool_issno'] = ~match.on_issno.isnull()

    # Count matches and categorize discrepancies for each scopus ID
    categs = pd.DataFrame.from_records(match.apply(_classify_ids, axis=1).values,
                                       columns=['n_vals', 'categ'],
                                       index=match.index)
    match = pd.concat([match, categs], axis=1)

    # Get 'final' UIDs based on ISSN combined > ISSN print > ISSN online > title
    final = match.on_issnp.where(match.on_issnc.isnull(), match.on_issnc)
    final = match.on_issno.where(final.isnull(), final)
    final = match.on_name.where(match.on_name_single & final.isnull(), final)
    match['uid'] = final
    # Get dictionary of NLM UIDs -> SCOPUS IDs
    temp = match['uid'].dropna()
    scop_dict = dict(zip(temp.values, temp.index.astype(str).values))

    if write_files:
        match.to_pickle(MATCH_PICKLE_PATH)
        with open(MATCH_JSON_PATH, 'w') as outfile:
            json.dump(scop_dict, outfile)
    return match


def _classify_ids(r):
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
    for var in ['issnc', 'issnp', 'issno']:
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


def _unused_get_pubmed_mapping(pub, scop):

    testt = pd.merge(pub[pub.is_unique_title_safe], scop[scop.is_unique_title_safe],
                     on='title_safe', how='inner', suffixes=['_pub', '_scop'])
    testt.rename(columns={'title_safe': 'title_safe_scop'}, inplace=True)

    testp = pd.merge(
        pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn_print']),
        scop.dropna(axis=0, subset=['issn_print']),
        on='issn_print', how='inner', suffixes=['_pub', '_scop'])

    testo = pd.merge(
        pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn_online']),
        scop.dropna(axis=0, subset=['issn_online']),
        on='issn_online', how='inner', suffixes=['_pub', '_scop'])

    testc = pd.merge(
        pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn_comb']),
        scop.dropna(axis=0, subset=['issn_comb']),
        on='issn_comb', how='inner', suffixes=['_pub', '_scop'])

    # test1 = pd.merge(
    #     pub[pub.unique_isoabbr_nodots].dropna(axis=0, subset=['issn1']),
    #     scop.dropna(axis=0, subset=['issn1']),
    #     on='issn1', how='inner', suffixes=['_pub', '_scop'])

    safe_to_abbr_dict = pub[pub.is_unique_title_safe & pub.unique_isoabbr_nodots] \
        .set_index('title_safe')['isoabbr_nodots'].to_dict()

    ambiguous_abbrvs = set()

    for merged, method in [(testc, 'issn_combined'),
                           (testo, 'issn_print'),
                           (testp, 'issn_online'),
                           (testt, 'safe_title'),
                           ]:
        print(f"Linking Scopus to Pubmed table via {method}")
        updates = merged.set_index('title_safe_scop')['isoabbr_nodots'].to_dict()
        n_additions = 0
        for key in updates:
            if key in safe_to_abbr_dict:
                # Handle contradictory update
                if safe_to_abbr_dict[key] != updates[key]:
                    print(f"CLASH {key!r}: PUBMED {safe_to_abbr_dict[key]} | SCOPUS: {updates[key]}")
                    ambiguous_abbrvs.add(key)
            else:
                n_additions += 1
                safe_to_abbr_dict[key] = updates[key]
        print(f"{n_additions} via {method}.\n")
