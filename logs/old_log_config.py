import functools

from common.constants import DATA_DIR
from common.helpers import get_date


IS_LOGGED = False


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
                    if result is not None:
                        message += f'\tresult={repr(result)}'
                    logger_.log(level, message.rstrip())

            return result

        return wrapped

    return wrapper


def record_debug_message(*args, **kwargs):
    if IS_LOGGED:
        logger.opt(lazy=True).debug(*args, **kwargs)


def record_info_message(*args, **kwargs):
    print(*args, **kwargs)
    if IS_LOGGED:
        logger.opt(lazy=True).info(*args, **kwargs)


if IS_LOGGED:
    logger.remove()
    logger.add(
        str(
            DATA_DIR
            / get_date()
            / f'log_{get_date()}.log'
        ),
        format="{time:YYYY-MM-DD HH:mm:ss,SSS} | {level} | {message}",
        level="INFO",
        rotation='00:00',
        backtrace=True,
        diagnose=True,
    )
