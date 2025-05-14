import inspect
import logging
import os

from termcolor import colored


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "magenta",
    }

    def format(self, record):
        # If custom caller info is provided, use it
        if hasattr(record, "custom_filename") and hasattr(record, "custom_lineno"):
            record.filename = record.custom_filename
            record.lineno = record.custom_lineno

        # all line in log will be colored
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname, "white"))

        # only log level will be colored
        # levelname_colored = colored(record.levelname, self.COLORS.get(record.levelname, 'white'))
        # record.levelname = levelname_colored
        # return super().format(record)

        # only keywords will be colored
        # message = record.msg
        # for word, color in self.KEYWORDS.items():
        #     if word in message:
        #         message = message.replace(word, colored(word, color))
        # record.msg = message
        # return super().format(record)


# config log
dev_logger = logging.getLogger("dev")
dev_formatter = ColoredFormatter(
    "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
dev_handler = logging.StreamHandler()
# set console output formatter
dev_handler.setFormatter(dev_formatter)
dev_logger.addHandler(dev_handler)

# set file output formatter
log_file = os.environ.get("RBASE_LOG_FILE", "")
if log_file:
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(file_formatter)
    dev_logger.addHandler(file_handler)
dev_logger.setLevel(logging.INFO)
dev_logger.propagate = False

progress_logger = logging.getLogger("progress")
progress_handler = logging.StreamHandler()
progress_handler.setFormatter(ColoredFormatter("%(message)s"))
progress_logger.addHandler(progress_handler)
if log_file:
    progress_logger.addHandler(file_handler)
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False

dev_mode = False


def set_dev_mode(mode: bool):
    """Set development mode"""
    global dev_mode
    dev_mode = mode


def set_level(level):
    """Set logging level"""
    dev_logger.setLevel(level)


def debug(message):
    """Debug log

    Records debug information and captures the caller's filename and line number
    """
    if dev_mode:
        # Get caller frame info
        caller = inspect.currentframe().f_back
        # Use custom_filename and custom_lineno to avoid overwriting built-in LogRecord attributes
        dev_logger.debug(
            message,
            extra={
                "custom_filename": os.path.basename(caller.f_code.co_filename),
                "custom_lineno": caller.f_lineno,
            },
        )


def info(message):
    """Info log

    Records general information and captures the caller's filename and line number
    """
    if dev_mode:
        caller = inspect.currentframe().f_back
        dev_logger.info(
            message,
            extra={
                "custom_filename": os.path.basename(caller.f_code.co_filename),
                "custom_lineno": caller.f_lineno,
            },
        )


def warning(message):
    """Warning log

    Records warning information and captures the caller's filename and line number
    """
    if dev_mode:
        caller = inspect.currentframe().f_back
        dev_logger.warning(
            message,
            extra={
                "custom_filename": os.path.basename(caller.f_code.co_filename),
                "custom_lineno": caller.f_lineno,
            },
        )


def error(message):
    """Error log

    Records error information and captures the caller's filename and line number
    """
    if dev_mode:
        caller = inspect.currentframe().f_back
        dev_logger.error(
            message,
            extra={
                "custom_filename": os.path.basename(caller.f_code.co_filename),
                "custom_lineno": caller.f_lineno,
            },
            exc_info=True,
        )


def critical(message):
    """Critical error log

    Records critical error information, captures the caller's filename and line number,
    then raises a runtime exception
    """
    caller = inspect.currentframe().f_back
    dev_logger.critical(
        message,
        extra={
            "custom_filename": os.path.basename(caller.f_code.co_filename),
            "custom_lineno": caller.f_lineno,
        },
    )
    raise RuntimeError(message)


def color_print(message, **kwargs):
    """Print colored information"""
    progress_logger.info(message)


def color_print_debug(message, **kwargs):
    """Print colored debug information"""
    progress_logger.debug(message)
