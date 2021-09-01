"""
Run app with `flask run`
or
`gunicorn -b 0.0.0.0:5005 -w 4 src.journal_targeter.journals:app`
"""
import os
import sys
import logging
import pathlib
import subprocess
from typing import Union

import click
import dotenv
from flask import render_template
from flask.cli import FlaskGroup
from flask_migrate import Migrate

from . import paths
from .app import create_app, db
from .app.models import Source
from .admin import copy_initial_data
from .reference import init_reference_data_from_cache

_APP_LOCATION = 'journal_targeter.journals:app'
_logger = logging.getLogger(__name__)
app = create_app(os.getenv('FLASK_CONFIG') or 'default')
migrate = Migrate(app, db)
copy_initial_data(app)


def init_data(init_refs=False, init_demo=False):
    if init_refs:
        with app.app_context():
            init_reference_data_from_cache()
    if init_demo:
        from .demo import init_demo
        init_demo(app.config['DEMO_PREFIX'], overwrite=False)


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
@click.pass_context
def cli(ctx: click.core.Context, verbose):
    if verbose:
        _logger.setLevel("DEBUG")
    # ref init process for update-sources and gunicorn happens elsewhere
    if ctx.invoked_subcommand not in {'setup', 'update-sources', 'gunicorn'}:
        init_data(init_refs=True)
    if ctx.invoked_subcommand in {'flask', 'gunicorn'}:
        init_data(init_demo=True)


@cli.group()
def setup():
    """Set up environment variables."""


@setup.command('prompt')
def config_prompt():
    """Create configuration .env file via command prompt."""
    env_dict = dict()
    prev_vals = dotenv.dotenv_values(paths.ENV_PATH).copy()

    def _set_env_via_prompt(var_name, prompt_str, default=None, **prompt_kw):
        if var_name in prev_vals:
            default = prev_vals[var_name]
        env_dict[var_name] = click.prompt(prompt_str, default=default, **prompt_kw)

    add_api_key = click.confirm("Store an NCBI API key?", default=False)
    if add_api_key:
        _set_env_via_prompt('API_KEY', 'NCBI API KEY')
    _set_env_via_prompt('FLASK_ENV', 'Environment', default='production',
                        type=click.Choice(['development', 'production']))
    _set_env_via_prompt('SECRET_KEY', 'Secret key (for encryption)',
                        default=os.urandom(16).hex())

    # Mail logger
    setup_mail = click.confirm("Set up error logging to email?", default=False)
    if setup_mail:
        _set_env_via_prompt('MAIL_SERVER', 'Mail server (e.g. mail.example.com)')
        _set_env_via_prompt('MAIL_SENDER', 'From address (e.g. Jot <info@example.com>)')
        _set_env_via_prompt('MAIL_USERNAME', 'Mail username', default='')
        _set_env_via_prompt('MAIL_PASSWORD', 'Mail password', default='')
        _set_env_via_prompt('MAIL_PORT', 'Mail port', default=25, type=int)
        _set_env_via_prompt('MAIL_USE_TLS', 'Use TLS', default=False)
        _set_env_via_prompt('MAIL_ADMIN', 'Alert address (e.g. admin@example.com)')

    if os.path.exists(paths.ENV_PATH):
        msg = "OK to overwrite previous .env file with these values?"
    else:
        msg = "OK to create an .env file with these values?"
    is_happy = click.confirm(msg, default=True)
    if not is_happy:
        click.echo("Aborting .env file creation.")
        return
    backup_path = _store_new_env(env_dict)
    if backup_path:
        click.echo(f"Previous config file saved to {backup_path}")


@setup.command('revert')
def config_revert():
    """Revert to previous .env file."""
    prev_path = _get_previous_env_path()
    if os.path.exists(prev_path):
        overwrite = click.confirm("Are you sure you want to restore previous env file?")
        if overwrite:
            os.rename(prev_path, paths.ENV_PATH)
            click.echo("Configuration reset to previous values.")
        else:
            click.echo("Aborted.")
    else:
        click.echo("No previous configuration file found.")


@setup.command('show')
def config_show():
    """Print configuration path and contents."""
    if paths.ENV_PATH.exists():
        click.echo(f"ENV PATH: {paths.ENV_PATH}\nContents:")
        with open(paths.ENV_PATH, 'r') as env:
            for line in env:
                click.secho(line.strip(), fg='green')
    else:
        click.echo("No configuration file found.")


@setup.command('edit')
def config_edit():
    """Open configuration .env file in an editor."""
    if not paths.ENV_PATH.exists():
        paths.ENV_PATH.touch()
    click.echo(f"Opening {paths.ENV_PATH}. Edit, then save when you're done.")
    click.launch(str(paths.ENV_PATH))


@cli.command()
def build_demo():
    """Create pubmed pickle object if needed; rebuild demo data."""
    from .demo import init_demo
    with app.app_context():
        demo_prefix = app.config['DEMO_PREFIX']
    init_demo(demo_prefix)


@cli.command()
@click.option('-y', '--yaml', 'query_yaml',
              type=click.Path(exists=True),
              help='Path to YAML file with title and abstract fields.')
@click.option('-r', '--ris', 'ris_path', type=click.Path(exists=True),
              help='Path to references file in RIS format.')
@click.option('-o', '--out_basename')
def match(query_yaml=None, ris_path=None, out_basename=None):
    """Run search and save html file."""
    if query_yaml is None:
        example_path = os.path.join(paths.DATA_ROOT, 'demo', 'example.yaml')
        import pathlib
        example_txt = pathlib.Path(example_path).read_text()
        query_yaml = click.edit(example_txt)
        if not query_yaml:
            click.secho("No YAML data provided.")
    if ris_path is None:
        ris_path = click.prompt('RIS path', type=click.Path(exists=True))
    if out_basename is None:
        out_basename = click.prompt('Output basename, e.g. <basename>.html',
                                    default='output')
    return match_data(query_yaml=query_yaml, ris_path=ris_path, out_basename=out_basename)


def create_app_cli():
    env_name = os.environ.get('FLASK_ENV', 'production')
    return create_app(env_name)


@cli.group(cls=FlaskGroup, create_app=create_app_cli)
def flask():
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
        _logger.info('API key present.') if api_key else _logger.info('API key absent.')
        medline_updated = pubmed.download_and_compare_pubmed_reference()
        tm_pickle_exists = os.path.exists(paths.TM_PICKLE_PATH)
        if medline_updated or not tm_pickle_exists:
            with app.app_context():
                pm_full = pubmed.load_pubmed_journals(api_key=api_key)
            tm = pubmed.TitleMatcher().init_data(pm_full)  # type: pubmed.TitleMatcher
            tm.save_pickle()
        else:
            _logger.info("No changes found in Medline journals list.")
    if scopus_path or jcr_path:
        # initialize TitleMatcher data
        from .reference import TM
        with app.app_context():
            TM.init_data()
    if scopus_path:
        from journal_targeter import scopus
        from journal_targeter.models import RefTable, TableMatcher
        scopsm = scopus.load_scopus_journals_reduced(scopus_path)  # 31 s
        scop = RefTable(df=scopsm, source_name='scopus', index_is_uid=True,
                        rename_dict={'citescore': 'CiteScore'},
                        title_col='journal_name', issn_print='Print-ISSN',
                        issn_online='E-ISSN', col_metrics=['CiteScore'],
                        col_other=['is_open'])
        tm_scop = TableMatcher(scop)
        tm_scop.match_missing(n_processes=ncpus, save=True)
        with app.app_context():
            Source.updated_now('scopus')
    if jcr_path:
        from journal_targeter.helpers import load_jcr_json
        from journal_targeter.models import RefTable, TableMatcher
        jif = load_jcr_json(jcr_path)
        jcr_ref = RefTable(source_name='jcr', df=jif, title_col='journal',
                           col_metrics=['JIF', 'JCI', 'AI', 'EF', 'EFn'],
                           issn_print='issn_print', issn_online='issn_online',
                           rename_dict={},
                           index_is_uid=False)
        jcr_tm = TableMatcher(jcr_ref)
        jcr_tm.match_missing(n_processes=ncpus, save=True)
        with app.app_context():
            Source.updated_now('jcr')


@cli.command(context_settings=dict(ignore_unknown_options=True,))
@click.argument('gunicorn_args', nargs=-1, type=click.UNPROCESSED)
def gunicorn(gunicorn_args):
    """Serve using gunicorn."""
    gunicorn_path = pathlib.Path(sys.executable).parent.joinpath('gunicorn')
    if not gunicorn_path.exists():
        click.secho("gunicorn not found.", fg='red')
        return
    cmdline = [str(gunicorn_path), ] + list(gunicorn_args) + [_APP_LOCATION]
    click.echo(f"Invoking: {' '.join(cmdline)}")
    subprocess.call(cmdline)


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


@app.shell_context_processor
def make_shell_context():
    from .reference import MT, TM
    return dict(MT=MT, TM=TM)


@app.cli.command()
@click.option("-y", "--yaml_path", help="YAML path with title and abstract.")
@click.option("-r", "--ris_path", help="RIS path with references.")
@click.option("--prefix", help="Name prefix for demo data, e.g. lung.")
def demo(yaml_path=None, ris_path=None, prefix=None):
    from .demo import create_demo_data_from_yaml
    create_demo_data_from_yaml(yaml_path, ris_path, prefix=prefix)


def _store_new_env(env_dict) -> Union[None, os.PathLike]:
    backup_path = None
    env_path_pl = pathlib.Path(paths.ENV_PATH)
    if env_path_pl.exists():
        backup_path = _get_previous_env_path()
        env_path_pl.rename(backup_path)
    env_path_pl.parent.mkdir(exist_ok=True)
    env_path_pl.touch()
    for key in env_dict:
        dotenv.set_key(env_path_pl, key, str(env_dict[key]), quote_mode='auto')
    _logger.info("Created .env file.")
    return backup_path


def _get_previous_env_path():
    backup_name = os.path.basename(paths.ENV_PATH.name) + '.prev'
    return os.path.join(paths.CONFIG_DIR, backup_name)


def _app_requires_data_init():
    """Check if app is being started, which requires proactive data init."""
    args = sys.argv[1:]
    if len(args) and args[0] in {'run'} or _APP_LOCATION in args:
        return True
    return False


if _app_requires_data_init():
    init_data(init_refs=True, init_demo=True)


if __name__ == '__main__':
    cli(['build-demo'])
