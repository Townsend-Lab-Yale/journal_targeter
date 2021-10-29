"""Data consolidation."""
import re
import logging
from functools import partial
from collections import OrderedDict

import pandas as pd

from .ref_loading import identify_user_references
from .lookup import lookup_jane
from .reference import MT, TM

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
    if query_title is None or query_abstract is None:
        raise ValueError("Provide query abstract, title and reference info.")
    elif ris_path is not None and refs_df is not None:
        raise ValueError("Provide ris_path or refs_df, not both.")

    if ris_path is None:
        refs_kw = dict() if refs_df is None else dict(refs_df=refs_df)
    else:
        refs_kw = dict(ris_path=ris_path)
    journals_t, articles_t, refs_df = process_inputs(
        input_text=query_title, input_type='TITLE', **refs_kw)
    journals_a, articles_a, _ = process_inputs(
        input_text=query_abstract, input_type='ABSTRACT', refs_df=refs_df)

    jf, af = aggregate_jane_journals_articles(journals_t, journals_a,
                                              articles_t, articles_a)
    if refs_df is not None:
        _add_jids_names_to_refs(refs_df, jf)  # finalize refs table
        # COMBINE CITED JOURNALS WITH JANE JOURNALS (-> JFM)
        citations = refs_df[['jid', 'refs_name', 'uid']].value_counts() \
            .reset_index(level=[1, 2]).rename(
                columns={0: 'cited'})
        jfm = jf.join(citations, how='outer', lsuffix='_jane', rsuffix='_refs')
        jfm.insert(0, 'uid', jfm.uid_refs.where(jfm.uid_jane.isnull(), jfm.uid_jane))
        jfm['cited'] = jfm['cited'].fillna(0).astype(int)
    else:
        jfm = jf.copy()
        jfm['refs_name'] = None
        jfm['cited'] = 0

    # Add metrics and main title
    jfm = _add_metrics_and_main_title(jfm)

    # ADD CAT
    zero_fill_cols = ['title_only', 'abstract_only', 'title', 'abstract', 'both']
    # jfm.fillna({i: 0 for i in zero_fill_cols})
    for col in zero_fill_cols:
        jfm[col] = jfm[col].fillna(0).astype(int)
    jfm['CAT'] = jfm['cited'] + jfm['abstract'] + jfm['title']

    jfm = jfm.sort_values(['CAT', 'sim_sum'], ascending=False).reset_index()  # type: pd.DataFrame

    # Add journal_name, using shortest name option
    jfm.insert(1, 'journal_name', jfm.apply(lambda r: _pick_short_journal_name(
        [r['jane_name'], r['refs_name'], r['main_title']]), axis=1))

    # Add short abbreviation for journals
    abbrv = jfm['uid'].map(MT.df['abbr'])
    jfm['abbr'] = abbrv.where(~abbrv.isnull(), jfm['journal_name'])
    jfm['abbr'] = jfm['abbr'].apply(lambda v: _get_short_str(v))
    # use MT is_open status, ignoring Jane 'is_oa'
    jfm['is_open'] = jfm['uid'].map(MT.df['is_open'])
    jfm.drop(columns=['is_oa'], inplace=True)
    # Fill in_medline from pm (partial via Jane)
    in_medline = jfm['uid'].map(MT.df['in_medline'])
    in_medline = in_medline.where(jfm['in_medline'].isnull(), jfm['in_medline'])
    jfm['in_medline'] = in_medline

    # Fill sr_id
    jfm['sr_id'] = jfm['uid'].map(MT.df['sr_id'])
    # Fill DOAJ data
    jfm['url_doaj'] = jfm['uid'].map(MT.df['url_doaj'])
    use_doaj_cols = ['url_doaj', 'license', 'author_copyright', 'n_weeks_avg',
                     'apc', 'apc_val', 'preservation', 'doaj_compliant',
                     'doaj_seal', 'doaj_score']
    for col in use_doaj_cols:
        jfm[col] = jfm['uid'].map(MT.df[col])

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
    # temp['conf_sum'] = temp['confidence']  # placeholder for conf sum aggregation
    groupby_col = 'jid'
    get_first = ['jane_name', 'influence', 'tags', 'uid', 'is_oa']
    # get_first += list(metrics.METRIC_NAMES)
    if from_api:
        get_first.extend(['in_medline', 'in_pmc'])
    get_sum = ['sim_sum']
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


def process_inputs(input_text=None, ris_path=None, refs_df=None, input_type=None):
    """Get similar journals and perform matching (user<>pubmed; JANE<>pubmed).

    Args:
        input_text (str): Text input for Jane.
        ris_path (path, optional): location of RIS references file.
        refs_df (pd.DataFrame): processed references (to pass through).
        input_type (str): description for logging, e.g. 'abstract' or 'title'

    Returns:
        processed DataFrames of journals, articles, and references.
    """

    # Process ris_path user citations if refs_df not provided
    if refs_df is None and ris_path is not None:
        refs_df = identify_user_references(ris_path)

    # Get matches and scores
    _logger.info(f"Running Jane search of {input_type or 'input'} text...")
    journals, articles = lookup_jane(input_text)
    # Add pubmed matching info
    # match_jane = TM.match_titles(journals.jane_name)  # orig used titles only
    match_jane = TM.lookup_uids_from_title_issn(titles=journals.jane_name,
                                                issn_print=journals.jane_issn)
    match_jane.index.name = 'j_rank'
    journals = pd.concat([journals, match_jane], axis=1, ignore_index=False)

    return journals, articles, refs_df


def _add_jids_names_to_refs(refs_df, jf):
    """Add columns to refs_df (jid, refs_name)."""
    jid_from_uid_dict = {uid: jid for jid, uid in jf['uid'].items() if uid != tuple()}
    jid_from_name_dict = {name: jid for jid, name in jf.loc[jf['uid'] == tuple(), 'jane_name'].items()}
    jid_matches = refs_df['uid'].map(jid_from_uid_dict)
    jid_matches = jid_matches.where(~jid_matches.isnull(), refs_df['user_journal'].map(jid_from_name_dict))
    jid_dict = jid_matches.dropna().to_dict()

    # For remaining no-jid refs, use uid if present
    needs_jid = refs_df.loc[jid_matches.isnull(), 'uid']  # ref_index: uid
    uids_without_jid = needs_jid.loc[lambda v: v != tuple()]
    jid_dict.update(uids_without_jid.apply(lambda v: f'u{v}').to_dict())

    # For remaining no-jid refs, use row index of first instance of name
    inds_no_uid_or_jid = needs_jid.loc[lambda v: v == tuple()].index
    if len(inds_no_uid_or_jid):
        extras = refs_df.loc[inds_no_uid_or_jid].user_journal
        first_inds = extras.drop_duplicates()
        extra_name_dict = {title: f"r{ind}" for ind, title in first_inds.items()}
        jid_dict.update(extras.map(extra_name_dict).to_dict())

    refs_df['jid'] = refs_df.index.map(jid_dict)

    # Get single refs journal name per jid
    name_counts = refs_df['user_journal'].value_counts()
    refs_df['name_counts'] = refs_df['user_journal'].map(name_counts)
    single_names = refs_df.sort_values(['jid', 'name_counts', 'user_journal'],
                                       ascending=[True, False, True])[
        ['jid', 'user_journal']].drop_duplicates(subset='jid') \
        .set_index('jid')['user_journal']
    refs_df['refs_name'] = refs_df['jid'].map(single_names)


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
    use_pm_cols = ['main_title'] + [i for i in MT.metric_list if i != 'influence']
    out = df.join(MT.df[use_pm_cols], how='left', on='uid')
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
    od.update(g[['jid', 'title', 'authors', 'year', 'a_id']]
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
