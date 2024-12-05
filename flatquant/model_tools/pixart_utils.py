import logging

import torch
import torch.nn as nn

from flatquant.utils import skip_initialization

logger = logging.getLogger(__name__)


def apply_flatquant_to_pixart(args, model):
    skip_initialization()
    logger.warning("TODO flatquant for pixart is not implemented")
    return model
