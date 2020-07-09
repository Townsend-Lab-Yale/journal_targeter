from flask import Blueprint

main = Blueprint('main', __name__)

from . import views, errors


@main.app_context_processor
def add_bokeh_version():
    import bokeh as bk
    return dict(bokeh_version=bk.__version__)
