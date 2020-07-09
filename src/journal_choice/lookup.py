import re
import sys
import unicodedata
from collections import OrderedDict

import bs4
import requests
import numpy as np
import pandas as pd

import logging
logging.basicConfig(format='%(levelname)s: %(message)s',  # %(asctime)-15s
                    level=logging.INFO, stream=sys.stdout)

JANE_URL = "https://jane.biosemantics.org/suggestions.php"


def lookup_jane(text=None):
    """Get journals and articles associated with query text.

    Args:
        text (str): Title or abstract text for Jane lookup.

    Returns:
        journals (pd.DataFrame): table of journals, ordered by rank
        articles (pd.DataFrame): table of articles, includes journal rank column.
    """
    if text is None:
        raise TypeError("Query text is required for Jane search.")
    form_dict = {'text': text,
                 'languageCount': 7,
                 'typeCount': 19,
                 'openaccess': 'no preference',
                 'pubmedcentral': 'no preference',
                 'findJournals': 'Find journals'}

    inf_template = r'Article Influence = [<]?([\d\.]+). Of all journals, (\d+)% ' \
                   'has a lower Article Influence score.'
    sim_template = r'Similarity to your query is (\d+)%'

    r = requests.post(JANE_URL, data=form_dict)
    soup = bs4.BeautifulSoup(r.content, features="lxml")
    table = soup.find('table')
    trs = table.find_all('tr', recursive=False)
    tr_j = trs[0::2]  # journal rows
    tr_a = trs[1::2]  # article rows

    # PARSE JOURNALS
    journal_list = []
    for tr in tr_j:
        confidence = re.findall(r'Confidence is (\d+)%', tr.td.div['title'])[0]
        j_cell = tr.find_all('td')[1]
        j_cell_text = j_cell.text.strip()
        tags = [i.text for i in j_cell.find_all('div')]
        if tags:
            first_tag_name = tags[0]
            journal_name = j_cell_text.split(first_tag_name)[0].strip()
        else:
            journal_name = j_cell_text.strip()

        inf_cell = tr.find_all('td')[-2]
        has_influence = inf_cell.div is not None
        if has_influence:
            inf_title = inf_cell.div.attrs['title']

            influence, pc_lower = re.match(inf_template, inf_title).groups()
        else:
            influence, pc_lower = np.nan, np.nan
        tags = [unicodedata.normalize('NFKD', i) for i in tags]
        row_dict = OrderedDict({'confidence': confidence,
                                'journal_name': journal_name,
                                'tags': '|'.join(tags),
                                'influence': influence,
                                'pc_lower': pc_lower
                                })
        journal_list.append(row_dict)
    journals = pd.DataFrame(journal_list)
    journals['confidence'] = journals['confidence'].astype(int)
    journals['influence'] = journals['influence'].astype(float)
    journals['pc_lower'] = journals['pc_lower'].astype(float)
    journals['is_oa'] = journals.tags.str.contains('open access')
    journals.index.name = 'j_rank'

    # PARSE ARTICLES
    a_list = []
    for j_rank, a_table in enumerate(tr_a):
        for tr in a_table.find_all('tr'):
            sim_str = tr.td.div.attrs['title']
            tds = tr.find_all('td')
            sim = re.match(sim_template, sim_str).groups()[0]
            checkbox = tds[1].input
            a_id = checkbox.attrs['name']
            url = tds[2].a.attrs['href']
            title = tds[2].strong.text
            meta = tds[2].text.split(title)
            authors = meta[0].strip()
            year = meta[1].split('.')[-1].strip()
            a_dict = OrderedDict({
                'j_rank': j_rank,
                'sim': sim,
                'title': title,
                'authors': authors,
                'year': year,
                'a_id': a_id,
                'url': url,
            })
            a_list.append(a_dict)
    articles = pd.DataFrame(a_list)
    articles['sim'] = articles['sim'].astype(int)

    # Get some aggregated journal/article info into journals table
    n_articles = articles.groupby('j_rank').size()
    sim_sum = articles.groupby('j_rank')['sim'].sum()
    sim_max = articles.groupby('j_rank')['sim'].max()
    sim_min = articles.groupby('j_rank')['sim'].min()
    sims = articles.groupby('j_rank')['sim'].apply(lambda s: '|'.join([str(i) for i in s]))
    meta = pd.concat([n_articles, sim_sum, sim_max, sim_min, sims], axis=1)
    meta.columns = ['n_articles', 'sim_sum', 'sim_max', 'sim_min', 'sims']
    journals = pd.concat([journals, meta], axis=1)
    col_order = ['journal_name', 'influence', 'n_articles', 'sim_sum', 'sim_max', 'pc_lower']
    col_order += [i for i in journals.columns if i not in col_order]
    journals = journals[col_order].rename(columns={'journal_name': 'jane_name'})
    return journals, articles
