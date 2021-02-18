"""
Run app with `flask run`
or
`gunicorn -b 0.0.0.0:5005 -w 4 src.journal_choice.journals:app`
"""
import os
import sys
import logging

import click
from flask import render_template
# from flask.cli import with_appcontext

from . import logger

from .app import create_app

app = create_app(os.getenv('FLASK_CONFIG') or 'default')


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
    else:
        handler = logger.handlers[0]
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.setLevel("WARNING")


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
