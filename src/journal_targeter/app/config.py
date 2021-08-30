import os
import logging

import dotenv

from .. import paths

basedir = os.path.abspath(os.path.dirname(__file__))
_logger = logging.getLogger(__name__)
dotenv.load_dotenv(paths.ENV_PATH, verbose=False)
# Ensure FLASK_APP is specified (in case served with flask)
if 'FLASK_APP' not in os.environ:
    os.environ['FLASK_APP'] = 'journal_targeter.journals'


class Config:
    DEMO_PREFIX = os.environ.get('DEMO_PREFIX', 'sars')
    API_KEY = os.environ.get('API_KEY')

    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(16)
    SESSION_TYPE = 'filesystem'
    SESSION_FILE_DIR = os.environ.get('SESSION_FILE_DIR', paths.SESSION_DIR)
    MAX_CONTENT_LENGTH = 1024 * 1024  # 1MB request size limit, for uploads

    SQLALCHEMY_DATABASE_URI = \
        'sqlite:///' + os.path.join(paths.DATA_ROOT, 'db.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.googlemail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in \
        ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_SUBJECT_PREFIX = '[Jot]'
    MAIL_SENDER = os.environ.get('MAIL_SENDER')
    MAIL_ADMIN = os.environ.get('MAIL_ADMIN')

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    @classmethod
    def init_app(cls, app):
        print("Running the production config init code.")
        Config.init_app(app)

        # email errors to the administrators
        import logging
        from logging.handlers import SMTPHandler
        credentials = None
        secure = None
        if getattr(cls, 'MAIL_USERNAME', None) is not None:
            credentials = (cls.MAIL_USERNAME, cls.MAIL_PASSWORD)
            if getattr(cls, 'MAIL_USE_TLS', None):
                secure = ()
        print(f"{cls.MAIL_SERVER=}, {cls.MAIL_ADMIN=}, {cls.MAIL_SERVER}")
        mail_handler = SMTPHandler(
            mailhost=(cls.MAIL_SERVER, cls.MAIL_PORT),
            fromaddr=cls.MAIL_SENDER,
            toaddrs=[cls.MAIL_ADMIN],
            subject=cls.MAIL_SUBJECT_PREFIX + ' Application Error',
            credentials=credentials,
            secure=secure)
        mail_handler.setLevel(logging.ERROR)
        app.logger.addHandler(mail_handler)


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}
