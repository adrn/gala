# Standard library
import logging
import sys


class CustomStreamHandler(logging.StreamHandler):
    def emit(self, record):
        record.origin = 'gala'

        if record.levelno <= logging.INFO:
            stream = sys.stdout
        else:
            stream = sys.stderr

        self.setStream(stream)
        super().emit(record)


class Logger(logging.getLoggerClass()):
    def _set_defaults(self):
        """Reset logger to its initial state"""

        # Remove all previous handlers
        for handler in self.handlers:
            self.removeHandler(handler)

        # Set default level
        self.setLevel(logging.INFO)

        # Set up the custom handler
        sh = CustomStreamHandler()

        # create formatter
        formatter = logging.Formatter('[%(origin)s] %(levelname)s: %(message)s')

        # add formatter to ch
        sh.setFormatter(formatter)

        self.addHandler(sh)


logging.setLoggerClass(Logger)
logger = logging.getLogger('gala')
logger._set_defaults()
