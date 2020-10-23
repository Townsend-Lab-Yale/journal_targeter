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
from flask.cli import with_appcontext


# LOGGING
LOG_LEVEL = getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(format='%(levelname)s: %(message)s',  # %(asctime)-15s
                    level=LOG_LEVEL, stream=sys.stdout)
_logger = logging.getLogger(__name__)

# ENSURE DOTENV VARIABLES HAVE LOADED (for gunicorn)
if not os.getenv('FLASK_CONFIG', ''):
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

from .app import create_app

app = create_app(os.getenv('FLASK_CONFIG') or 'default')


@app.cli.command('match')
@click.option('-y', '--yaml', 'query_yaml',
                required=True, type=click.Path(exists=True),
                help='Path to YAML file with title and abstract fields.')
@click.option('-r', '--ris', 'ris_path', type=click.Path(exists=True),
                help='Path to references file in RIS format.')
@click.option('-o', '--out_basename', default='out')
def match(query_yaml=None, ris_path=None, out_basename=None):
    from .mapping import run_queries
    from .plot import get_bokeh_components
    from .helpers import get_queries_from_yaml
    query_dict = get_queries_from_yaml(query_yaml)
    query_title = query_dict['title']
    query_abstract = query_dict['abstract']
    j, a, jf, af, refs_df = run_queries(query_title=query_title,
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
    print(f"Results written to {out_path}.")


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
@click.argument("yaml_path")
@click.option("--prefix", help="Name prefix for demo data, e.g. lung.")
def demo(yaml_path, prefix=None):
    from .demo import create_demo_data_from_yaml
    create_demo_data_from_yaml(yaml_path, prefix=prefix)
