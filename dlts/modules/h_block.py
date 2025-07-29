import torch.nn as nn

from .inverted_residual import InvertedResidual
from .lp_vit import LPViTBlock


class HBlock(nn.Module):
    def __init__(
            self
    ) -> None:
        super(HBlock, self).__init__()





