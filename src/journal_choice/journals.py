"""
Run app with `flask run`
or
`gunicorn -b 0.0.0.0:5005 -w 4 src.journal_choice.journals:app`
"""
import os
import sys
import logging

import click


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
