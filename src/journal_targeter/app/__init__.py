import nltk
from flask import Flask
from flask_bootstrap import Bootstrap
from flask_session import Session
from flaskext.markdown import Markdown
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from .config import config


bootstrap = Bootstrap()
session = Session()
db = SQLAlchemy(session_options={'expire_on_commit': False})
migrate = Migrate()


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    bootstrap.init_app(app)
    session.init_app(app)
    Markdown(app)
    db.init_app(app)
    migrate.init_app(app, db)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    @app.before_first_request
    def before_first_request():
        from ..demo import init_demo
        init_demo(app.config['DEMO_PREFIX'], overwrite=False)

    return app

