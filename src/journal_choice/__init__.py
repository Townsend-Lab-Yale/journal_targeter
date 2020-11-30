# -*- coding: utf-8 -*-
import os
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

DATA_DIR = os.path.sep.join(__file__.split(os.path.sep)[:-3] + ['data'])
DEMO_DIR = os.path.join(DATA_DIR, 'demo')
METRICS_DIR = os.path.join(DATA_DIR, 'metrics')
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
