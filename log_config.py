from datetime import date
import functools
from os.path import join

from constants import PATH_TO_DATA, IS_LOGGED


if IS_LOGGED:
    from loguru import logger
else:
    logger = None


def logger_wraps(*, is_entry=True, is_exit=True, level="DEBUG"):

    def wrapper(func):
        func_name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if IS_LOGGED:
                logger_ = logger.opt(depth=1)
                if is_entry:
                    message = f"Entering '{func_name}'\n"
                    if kwargs:
                        message += '\tkwargs:\n'
                        for key, value in kwargs.items():
                            message += f"\t\t{repr(key)}: {repr(value)}\n"
                    logger_.log(level, message.rstrip())
            result = func(*args, **kwargs)
            if IS_LOGGED:
                logger_ = logger.opt(depth=1)
                if is_exit:
                    message = f"Exiting '{func_name}'\n"
                    if result:
                        message += f'\tresult={repr(result)}'
                    logger_.log(level, message.rstrip())

            return result

        return wrapped

    return wrapper


def debug_info(*args, **kwargs):
    if IS_LOGGED:
        logger.opt(lazy=True).debug(*args, **kwargs)


if IS_LOGGED:
    logger.remove()
    logger.add(
        join(PATH_TO_DATA, f'log_{date.today()}.log'),
        format="{time:YYYY-MM-dd HH:mm:ss,SSS} | {level} | {message}",
        level="DEBUG",
        rotation='00:00',
        backtrace=True,
        diagnose=True,
    )
