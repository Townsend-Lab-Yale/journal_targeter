import pytest
import pandas as pd


from journal_targeter.demo import get_demo_data_with_prefix
# from journal_targeter.mapping import run_queries


@pytest.fixture()
def df_dict():
    data = get_demo_data_with_prefix('sars')
    return data


# def test_sim_aggregation(df_dict):
#     a = df_dict['a']
#     j = df_dict['j']
#     af = df_dict['af']
#     jf = df_dict['jf']
#
#     # j, a article count matches
#     jn = j.groupby(['source', 'j_rank'])['n_articles'].first()
#     an = a.groupby(['source', 'j_rank'])['jid'].apply(len)
#     assert jn.equals(an)
#
#     # only one jid per source/rank
#     assert (j.groupby(['source', 'j_rank'])['jid'].apply(pd.Series.nunique) == 1).all()
#
#     # maxs match a, j
#     amax = a.groupby(['jid', 'source'])['sim'].max()
#     jmax = j.groupby(['jid', 'source'])['sim_max'].first()
#     assert (jmax - amax == 0).all()
#
#     # jf: one row per jid
#     assert (jf.groupby('jid').apply(len) == 1).all()
#
#     # mins match a, jf
#     amin = a.groupby('jid')['sim'].min()
#     jfmin = jf.set_index('jid')['sim_min']
#     assert ((amin - jfmin) == 0).all()
#
#     # maxs match j, jf
#     jmaxmax = jmax.groupby('jid').max()
#     jfmax = jf.groupby('jid')['sim_max'].first()
#     assert ((jfmax - jmaxmax) == 0).all()
#
#     # af sim extremes agree with jf sim limits
#     # note jf sim_min is lowest observed article sim (not lowest max per article)
#     alim = af.groupby('jid')['sim_max'].aggregate(['max', 'min'])
#     assert ((alim['min'] - jf.set_index('jid')['sim_min']) >= 0).all()
#     assert ((alim['max'] - jf.set_index('jid')['sim_max']) == 0).all()
