import os
import json
import multiprocessing
from collections import defaultdict

import pandas as pd

from . import scopus, pubmed, DATA_DIR
# from .pubmed import TitleMatcher

MATCH_PICKLE_PATH = os.path.join(DATA_DIR, 'match.pickle.gz')
MATCH_JSON_PATH = os.path.join(DATA_DIR, 'scopus_match.json')


def load_scopus_map():
    """Get dictionary of NLM UID -> Scopus ID, based on build_uid_match_table."""
    with open(MATCH_JSON_PATH) as infile:
        scopus_id_dict = json.load(infile)
    return scopus_id_dict


def build_uid_match_table(pm, tm, n_processes=4, write_files=False):
    """Get NLM UIDs for all Scopus journal IDs based on Scopus names and ISSNs.

    Args:
        pm (pd.DataFrame): table of pubmed info, built in pubmed.py.
        tm (journals.pubmed.TitleMatcher): used to match titles.
        n_processes (int): number of processes. Use mutiprocessing if >1.
        write_files (bool): write match table to pickle and json in DATA dir.
    """
    tm = pubmed.TitleMatcher(pm)
    scop = scopus.load_scopus_journals_reduced()
    use_multi = True if n_processes > 1 else False
    # Attempt match on Scopus journal titles
    titles = scop.journal_name.values
    if use_multi:
        with multiprocessing.Pool(processes=6) as pool:
            uid_list = list(pool.map(tm.get_uids_from_title, titles))
    else:
        uid_list = list(map(tm.get_uids_from_title, titles))
    match = pd.DataFrame.from_records(uid_list, columns=['on_name', 'name_method'])
    match = match.set_index(scop.index)
    is_single = (match.name_method != 'unmatched') & ~match.on_name.apply(lambda v: type(v) is tuple)
    match['on_name_single'] = is_single

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
