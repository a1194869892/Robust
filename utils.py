import math

import numpy as np
from sklearn.utils import shuffle
import logging
import os

def next_batch(x, batch_size):
    num_samples = x.shape[0]
    index = np.linspace(0, num_samples - 1, num_samples, dtype=int)
    index = shuffle(index)
    total = int(math.ceil(num_samples / batch_size))
    for i in range(total):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(num_samples, end_idx)
        idx = index[start_idx: end_idx]
        batch_x = x[idx]
        yield (batch_x, (i + 1))

def get_logger(root = './training_logs', filename = None):
    """
    Get logger.

    Parameters
    - root: str. Root directory of log files.
    - filename: str, Optional. The name of log files.

    return
    - logger: Logger
    """
    # logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
    fmt = '[%(asctime)s]%(levelname)s: %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S')

    if filename is not None:
        """Save logs as files"""
        if not os.path.exists(root):
            os.makedirs(root)

    # mode = 'w', overwriting the previous content, 'a', appended to previous file.
    fh = logging.FileHandler(os.path.join(root, filename), "a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    """Print logs at terminal"""
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger