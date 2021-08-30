import os
from flask import Blueprint, current_app

main = Blueprint('main', __name__)

from . import views, errors
from .. import models


@main.app_context_processor
def utility_processor():
    bokeh_version = _get_bokeh_version()
    return dict(bokeh_version=bokeh_version,
                get_static_text=_get_static_text)


def _get_bokeh_version():
    import bokeh as bk
    return bk.__version__


def _get_static_text(filename):
    fullpath = os.path.join(current_app.static_folder, filename)
    with open(fullpath, 'r') as f:
        return f.read()
