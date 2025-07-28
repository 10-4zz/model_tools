
import torch.nn as nn

from dlts.utils import Registry


LAYER_REGISTRY = Registry(
    registry_name="layer_registry",
    base_type=nn.Module,
)

