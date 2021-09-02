# adapted from https://github.com/park-project/park/blob/master/park/logger.py
import logging
from smart_kube.util.constants import (
    LOGGING_LEVEL,
    LOG_TO
)

if LOGGING_LEVEL == 'debug':
    level = logging.DEBUG
elif LOGGING_LEVEL == 'info':
    level = logging.INFO
elif LOGGING_LEVEL == 'warning':
    level = logging.WARNING
elif LOGGING_LEVEL == 'error':
    level = logging.ERROR
else:
    raise ValueError('Unknown logging level ' + LOGGING_LEVEL)


if LOG_TO == 'print':
    logging.basicConfig(level=level)
else:
    logging.basicConfig(filename=LOG_TO, level=level)
    # logging.basicConfig(format='%(message)s', filename=LOG_TO, level=level)


def debug(msg):
    logging.debug(msg)


def info(msg):
    logging.info(msg)


def warn(msg):
    logging.warning(msg)


def error(msg):
    logging.error(msg)


def exception(msg, *args, **kwargs):
    logging.exception(msg, *args, **kwargs)
