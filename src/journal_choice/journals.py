"""
Run app with `flask run`
"""
import os
import sys
import logging

# LOGGING
LOG_LEVEL = getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(format='%(levelname)s: %(message)s',  # %(asctime)-15s
                    level=LOG_LEVEL, stream=sys.stdout)
_logger = logging.getLogger(__name__)

from .app import create_app

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
