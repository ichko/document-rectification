import logging
import os
import sys

import torch


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

is_debug = "--debug" in sys.argv
if is_debug:
    os.environ["WANDB_MODE"] = "offline"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
