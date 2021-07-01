import os
import sys
import logging
from typing import Union


if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "journal_targeter"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


def _create_logger(log_level: Union[str, None] = None):
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level_int = getattr(logging, log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level_int)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)

    _logger = logging.getLogger(__name__)
    _logger.addHandler(ch)
    _logger.setLevel(log_level_int)

    return _logger


logger = _create_logger()
