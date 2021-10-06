import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


import os
import logging
from typing import Union

from rich.logging import RichHandler


def _create_logger(log_level: Union[str, None] = None):
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level_int = getattr(logging, log_level)

    ch = RichHandler(rich_tracebacks=True, show_level=False, show_path=False,
                     locals_max_string=120)
    ch.setLevel(log_level_int)
    formatter = logging.Formatter(fmt='%(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)

    _logger = logging.getLogger(__name__)
    _logger.addHandler(ch)
    _logger.setLevel(log_level_int)

    return _logger


logger = _create_logger()
