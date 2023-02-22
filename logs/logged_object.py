"""The module contains the base class for an instance that supports logging."""


from logs.log_config import CustomLogger

from pretty_repr import RepresentableObject


class LoggedObject(RepresentableObject):
    """
    Base class for an instance that supports logging.

    Methods
    -------
    process_exception(exception, description)
        Log and return exception message.

    """

    @property
    def logger_type(self) -> str:
        """Type field of `logs.CustomLogger` instance."""
        return self.__class__.__name__

    @property
    def logger(self) -> CustomLogger:
        """A custom logger."""
        return CustomLogger(type_=self.logger_type)

    def process_exception(self, exception: Exception, description: str) -> str:
        """Log and return exception message."""
        exception_message = exception.__class__.__name__ + f': {description} '
        if str(exception):
            exception_message += str(exception)
        self.logger.type_ = 'Exception'
        self.logger.warning(exception_message)
        self.logger.type_ = self.logger_type
        return exception_message
