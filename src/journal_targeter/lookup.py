import re
import unicodedata
from collections import OrderedDict

import bs4
import requests
import numpy as np
import pandas as pd
import zeep

import logging

JANE_URL = "https://jane.biosemantics.org/suggestions.php"
WSDL_URL = "http://jane.biosemantics.org:8080/JaneServer/services/JaneSOAPServer?wsdl"
_logger = logging.getLogger(__name__)


class ApiClient:

    def __init__(self):
        self.url = WSDL_URL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            _logger.debug("Creating Jane API client connection.")
            self._client = zeep.Client(WSDL_URL)
        return self._client

    def get_journal_results(self, query_text):
        return self.client.service.getJournals(text=query_text)


CLIENT = ApiClient()


def lookup_jane(text=None):
    """Get journals and articles tables from Jane search inc computed stats.

    Args:
        text (str): Title or abstract text for Jane lookup.

    Returns:
        journals DataFrame, ordered by rank; articles Dataframe, including
                journal rank column.
    """
    if text is None:
        raise TypeError("Query text is required for Jane search.")

    journals, articles = fetch_jane_results_via_api(text=text)

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


def fetch_jane_results_via_api(text=None):
    """Get journal and article results via HTML scraping.

        Args:
            text (str): Title or abstract text for Jane lookup.

        Returns:
            journals DataFrame, ordered by rank; articles Dataframe, including
                journal rank column.
    """
    res = CLIENT.get_journal_results(query_text=text)

    # JOURNALS
    records = []
    for j in res:
        rec = dict()
        rec['journal_name'] = j['name']
        rec['confidence'] = j['score'] * 100  # type: float
        rec['is_oa'] = j['openAccess'] == 'true'
        rec['jane_abbr'] = j['journalAbbr']
        rec['jane_issn'] = j['issn']
        rec['in_medline'] = j['medlineIndexed'] == 'true'
        rec['influence'] = np.nan if j['ai'] == '-1' \
            else 0.05 if j['ai'] == '<0.1' \
            else float(j['ai'])
        rec['pc_lower'] = int(j['airank']) if j['airank'] is not None else np.nan
        rec['pmc_months'] = int(j['pmcMonths'])
        rec['in_pmc'] = rec['pmc_months'] > -1
        rec['tags'] = _assemble_tags_for_api(rec)
        records.append(rec)
    journals = pd.DataFrame.from_records(records)
    journals.index.name = 'j_rank'

    # ARTICLES
    records = []
    for j_rank, j in enumerate(res):
        papers = j['papers']
        for article in papers:
            rec = dict()
            rec['j_rank'] = j_rank
            rec['sim'] = article.score * 100
            authors = [str(i) for i in article.authors if i is not None]
            rec['authors'] = ','.join(authors)
            rec['a_id'] = f'PMID_{article.pmid}'
            rec['title'] = article.title
            rec['year'] = article.year
            # rec['url'] = f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}"
            records.append(rec)
    articles = pd.DataFrame.from_records(records)
    return journals, articles


def fetch_jane_results_via_scrape(text=None):
    """Get journal and article results via HTML scraping.

        Args:
            text (str): Title or abstract text for Jane lookup.

        Returns:
            journals DataFrame, ordered by rank; articles Dataframe, including
                journal rank column.
    """
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
    return journals, articles


def _assemble_tags_for_api(record):
    """Get helpful 'tags' string for journal info from API, mimicking Jane tags.
    >>> _assemble_tags_for_api({'is_oa': True, 'in_pmc': True, 'in_medline': False})
    'open access | PMC'
    >>> _assemble_tags_for_api({'is_oa': True, 'in_pmc': True, 'in_medline': True})
    'open access | medline-indexed | PMC'
    """
    vals = []
    if record['is_oa']:
        vals.append('open access')
    if record['in_medline']:
        vals.append('medline-indexed')
    if record['in_pmc']:
        vals.append('PMC')
    return ' | '.join(vals)
