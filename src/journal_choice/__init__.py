# -*- coding: utf-8 -*-
import os
import logging
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


# ENSURE DOTENV VARIABLES HAVE LOADED (for gunicorn)
if not os.getenv('FLASK_CONFIG', ''):
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())


def _create_logger():
    _logger = logging.getLogger(__name__)
    log_level = getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper())
    _logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    _logger.addHandler(ch)
    return _logger


logger = _create_logger()

DATA_DIR = os.path.sep.join(__file__.split(os.path.sep)[:-3] + ['data'])
DEMO_DIR = os.path.join(DATA_DIR, 'demo')
METRICS_DIR = os.path.join(DATA_DIR, 'metrics')
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
