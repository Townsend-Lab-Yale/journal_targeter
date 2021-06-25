"""
Run app with `flask run`
or
`gunicorn -b 0.0.0.0:5005 -w 4 src.journal_choice.journals:app`
"""
import os
import logging
import pathlib
from typing import Union

import click
import dotenv
from flask import Flask, render_template

from . import paths
from .app import create_app
from .admin import copy_initial_data, load_dotenv_as_dict


copy_initial_data()
app = create_app(os.getenv('FLASK_CONFIG') or 'default')
_logger = logging.getLogger(__name__)


@app.cli.command('match')
@click.option('-y', '--yaml', 'query_yaml',
              required=True, type=click.Path(exists=True),
              help='Path to YAML file with title and abstract fields.')
@click.option('-r', '--ris', 'ris_path', type=click.Path(exists=True),
              help='Path to references file in RIS format.')
@click.option('-o', '--out_basename', default='out')
def flask_match(**kwargs):
    """Run search and save html file."""
    return match_data(**kwargs)


@click.group()
@click.option('--verbose/--quiet', default=False)
def cli(verbose):
    if verbose:
        logger.setLevel("DEBUG")


@cli.command()
@click.option('-y', '--yaml', 'query_yaml',
              required=True, type=click.Path(exists=True),
              help='Path to YAML file with title and abstract fields.')
@click.option('-r', '--ris', 'ris_path', type=click.Path(exists=True),
              help='Path to references file in RIS format.')
@click.option('-o', '--out_basename', default='out')
def match(query_yaml=None, ris_path=None, out_basename=None):
    """Run search and save html file."""
    return match_data(query_yaml=query_yaml, ris_path=ris_path, out_basename=out_basename)


@cli.command()
def run(**kwargs):
    """Run the server."""
    app.run()


from flask.cli import FlaskGroup


def create_dev_app():
    return create_app('development')


@cli.group(cls=FlaskGroup, create_app=create_dev_app)
def runserver():
    """Serve using Flask cli."""


@cli.command()
@click.option('--update-nlm/--skip-nlm', default=True, show_default=True,
              help="Refresh pubmed data")
@click.option("-s", "--scopus_path", type=click.Path(exists=True),
              help="Scopus 'ext_list' XLSX file.")
@click.option("-j", "--jcr_path", type=click.Path(exists=True),
              help="JCR JSON file")
@click.option("-n", "--ncpus", type=int, default=1, show_default=True,
              help="Number of processes for parallel matching.")
def update_sources(update_nlm, scopus_path, jcr_path, ncpus):
    """Update data sources, inc NLM, Scopus and JCR."""
    if update_nlm:
        from . import pubmed
        api_key = app.config['API_KEY']
        if api_key:
            _logger.info('API key present')
        else:
            _logger.info('API key absent')
        medline_updated = pubmed.download_and_compare_pubmed_reference()
        tm_pickle_exists = os.path.exists(paths.TM_PICKLE_PATH)
        if medline_updated or not tm_pickle_exists:
            pm_full = pubmed.load_pubmed_journals(api_key=api_key)
            tm = pubmed.TitleMatcher().init_data(pm_full)  # type: pubmed.TitleMatcher
            tm.save_pickle()
        else:
            _logger.info("No changes found in Medline journals list.")
    if scopus_path or jcr_path:
        # initialize TitleMatcher data
        from .reference import TM
        TM.init_data()
    if scopus_path:
        from journal_choice import scopus
        from journal_choice.models import RefTable, TableMatcher
        scopsm = scopus.load_scopus_journals_reduced(scopus_path)  # 31 s
        scop = RefTable(df=scopsm, source_name='scopus', index_is_uid=True,
                        rename_dict={'citescore': 'CiteScore'},
                        title_col='journal_name', issn_print='Print-ISSN',
                        issn_online='E-ISSN', col_metrics=['CiteScore'],
                        col_other=['is_open'])
        tm_scop = TableMatcher(scop)
        tm_scop.match_missing(n_processes=ncpus, save=True)
    if jcr_path:
        from journal_choice.helpers import load_jcr_json
        from journal_choice.models import RefTable, TableMatcher
        jif = load_jcr_json(jcr_path)
        jcr_rename_dict = {'journalImpactFactor': 'Impact',
                           'eigenFactorScore': 'EF',
                           'articleInfluenceScore': 'AI',
                           'normEigenFactor': 'EFn', }
        jcr_ref = RefTable(source_name='jcr', df=jif, title_col='journalTitle',
                           col_metrics=['Impact', 'AI', 'EF', 'EFn'],
                           issn_col='issn', rename_dict=jcr_rename_dict,
                           index_is_uid=False)
        jcr_tm = TableMatcher(jcr_ref)
        jcr_tm.match_missing(n_processes=ncpus, save=True)


@cli.command()
def build():
    """Update data sources, inc NLM, Scopus and JCR."""
    from .reference import init_reference_data_from_cache
    init_reference_data_from_cache()
    from .demo import init_demo
    init_demo()


def match_data(query_yaml=None, ris_path=None, out_basename=None):
    """From search inputs, create results page and save to HTML file."""
    from .mapping import run_queries
    from .plot import get_bokeh_components
    from .helpers import get_queries_from_yaml
    query_dict = get_queries_from_yaml(query_yaml)
    query_title = query_dict['title']
    query_abstract = query_dict['abstract']
    jf, af, refs_df = run_queries(query_title=query_title,
                                  query_abstract=query_abstract,
                                  ris_path=ris_path)
    js, divs = get_bokeh_components(jf, af, refs_df)
    ris_name = os.path.basename(ris_path)
    with app.app_context():
        html = render_template('index.html',
                               standalone=True,
                               query_title=query_title,
                               query_abstract=query_abstract,
                               query_ris=ris_name,
                               bokeh_js=js,
                               bokeh_divs=divs,
                               )
    basename_safe = out_basename.replace(os.path.sep, '')
    out_path = f"{basename_safe}.html"
    with open(out_path, 'w') as out:
        out.write(html)
    click.echo(f"Results written to {out_path}.", color='green')


@app.cli.command()
def deploy():
    """Run deployment tasks."""
    import shutil
    from .demo import update_demo_plot
    update_demo_plot(os.environ.get('DEMO_PREFIX', 'demo'))
    use_env = '.env.server'
    shutil.copy(use_env, '.env')
    print(f"Copied from {use_env} to .env")


@app.cli.command()
def develop():
    """Set up development server."""
    import shutil
    use_env = '.env.dev'
    shutil.copy(use_env, '.env')
    print(f"Copied from {use_env} to .env")


@app.cli.command()
@click.option("-y", "--yaml_path", help="YAML path with title and abstract.")
@click.option("-r", "--ris_path", help="RIS path with references.")
@click.option("--prefix", help="Name prefix for demo data, e.g. lung.")
def demo(yaml_path=None, ris_path=None, prefix=None):
    from .demo import create_demo_data_from_yaml
    create_demo_data_from_yaml(yaml_path, ris_path, prefix=prefix)
