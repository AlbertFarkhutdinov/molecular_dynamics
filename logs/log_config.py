"""The module contains logging configurations."""


from datetime import datetime
import json
import logging.config
import logging
import time
import traceback
from typing import Union

from common.constants import BASE_DIR


STDOUT_LOGGING_LEVEL = 'INFO'
FILE_LOGGING_LEVEL = 'DEBUG'


LOG_FORMAT = json.dumps({
    'loggedAt': '%(asctime)s.%(msecs)03dZ',
    'level': '%(levelname)s',
    'type': '%(type_)s',
    'message': '%(message)s',
    'trace': '%(trace)s',
})

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': LOG_FORMAT,
            'datefmt': '%Y-%m-%dT%H:%M:%S',
        },
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'level': STDOUT_LOGGING_LEVEL,
            'formatter': 'default',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': FILE_LOGGING_LEVEL,
            'formatter': 'default',
            'filename': str(
                BASE_DIR.joinpath(
                    'logs',
                    f'{datetime.utcnow().date().isoformat()}.log',
                )
            ),
            'mode': 'a',
            'encoding': 'utf-8',
            'when': 'midnight',
            'interval': 1,
        },
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'propagate': True,
            'handlers': ['stdout', 'file'],
        }
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logging.Formatter.converter = time.gmtime


OLD_FACTORY = logging.getLogRecordFactory()
ExceptionMessage = Union[str, Exception]


class CustomLogger:
    """
    A custom logger for conveniently passing
    of contextual information into logging calls.

    Parameters
    ----------
    type_ : str
        Type field in a log message.

    Methods
    -------
    debug(msg, *args, **kwargs)
        Log a message with level 'DEBUG' on this logger.
    info(msg, *args, **kwargs)
        Log a message with level 'INFO' on this logger.
    warning(msg, *args, **kwargs)
        Log a message with level 'WARNING' on this logger.
    error(msg, *args, **kwargs)
        Log a message with level 'ERROR' on this logger.
    critical(msg, *args, **kwargs)
        Log a message with level 'CRITICAL' on this logger.
    init_info(instance, start_time)
        Log a message with level 'INFO' about successful initialization
        of an instance.

    """

    def __init__(self, type_: str = '') -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        self.type_ = type_ or 'Undefined type'
        logging.setLogRecordFactory(self.__record_factory)
        self.__logger = logging.getLogger(__name__)

    def __record_factory(self, *args, **kwargs) -> logging.LogRecord:
        """Return a callable which is used to create a log record."""
        record = OLD_FACTORY(*args, **kwargs)
        record.type_ = self.type_
        trace = repr(
            traceback.format_exc()
        ).replace('"', r'\"').replace(r'\'', r'\"')
        record.trace = trace if trace != r"'NoneType: None\n'" else ''
        return record

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a message with level 'DEBUG' on this logger."""
        self.__logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log a message with level 'INFO' on this logger."""
        self.__logger.info(msg, *args, **kwargs)

    def warning(self, msg: ExceptionMessage, *args, **kwargs) -> None:
        """Log a message with level 'WARNING' on this logger."""
        self.__logger.warning(msg, *args, **kwargs)

    def error(self, msg: ExceptionMessage, *args, **kwargs) -> None:
        """Log a message with level 'ERROR' on this logger."""
        self.__logger.error(msg, *args, **kwargs)

    def critical(self, msg: ExceptionMessage, *args, **kwargs) -> None:
        """Log a message with level 'CRITICAL' on this logger."""
        self.__logger.critical(msg, *args, **kwargs)
