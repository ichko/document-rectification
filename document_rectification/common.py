import logging
import sys


def get_logger():
    # SRC - https://stackoverflow.com/questions/533048/how-to-log-source-file-name-and-line-number-in-python
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s >> %(message)s @ %(module)s:%(lineno)d"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()
