import sys
import logging

class RedirectingLogger:
    """A class that optionally copies OCHRE print statements to 
        - /dev/null
        - a file
        - stdout

       Used in the decorator "redirect_print_statements".
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __enter__(self):
        """Redirect stdout to self """
        self.terminal = sys.stdout
        sys.stdout = self

    def __exit__(self, *args):
        """Restore stdout on exit"""
        sys.stdout = self.terminal

    def write(self, message):
        """Write message to logger"""
        if '' == message.rstrip():
            return
        if 'warning' in message.lower():
            # remove 'WARNING: ' from message
            message = message[9:]
            if message.rstrip() != '':
                self.logger.warning(message)
        elif 'error' in message.lower():
            self.logger.error(message)
        else:
            self.logger.info(message)

    def flush(self):
        self.logger.flush()


def redirect_print_statements(logger: logging.Logger):
    """A decorator that redirects print statements to a logger."""
    def redirect_print(fn):
        def wraps(*args, **kwargs):
            with RedirectingLogger(logger):
                return fn(*args, **kwargs)
        return wraps
    return redirect_print


def get_logger(name, handler_types=['stream'], log_file=None) -> logging.Logger:
    """Create a logger with one or more handler types.

    Args:
        handler_types (list): A list of handler types. Defaults to ['stream']. 
            Options include 'stream', 'file', and 'null'. If none are provided,
            a null handler is used.
        log_file (str): The path to the log file. Defaults to None.
    Returns:
        logging.Logger: The logger.
    """
    if len(handler_types) == 0:
        handler_types = ['null']
    assert all([handler_type in ['stream', 'file', 'null'] for handler_type in handler_types]), \
        'Invalid handler type. Options include "stream", "file", and "null".'
    if 'file' in handler_types:
        assert log_file is not None, 'If "file" is in handler_types, a log filename must be provided.'
    
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] %(message)s', '%H:%M:%S')

    logger.handlers = []
    for handler_type in handler_types:
        if handler_type == 'stream':
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if handler_type == 'file':
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        if handler_type == 'null':
            nh = logging.NullHandler()
            logger.addHandler(nh)
        
    # default to INFO, change elsewhere if needed
    logger.setLevel(logging.INFO)

    return logger