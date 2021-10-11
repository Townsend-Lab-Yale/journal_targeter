from importlib import resources

from flask import Blueprint

main = Blueprint('main', __name__)

from . import views, errors
from .. import models, source_tracker


@main.app_context_processor
def utility_processor():
    bokeh_version = _get_bokeh_version()
    return dict(bokeh_version=bokeh_version,
                source_dates=source_tracker.dates_user,
                get_static_text=_get_static_text)


def _get_bokeh_version():
    import bokeh as bk
    return bk.__version__


def _get_static_text(filename):
    with resources.path('journal_targeter.app.static', filename) as path:
        with open(path, 'r') as infile:
            return infile.read()
