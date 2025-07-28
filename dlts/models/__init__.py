import torch.nn as nn

import model_logger_dp as logger

from dlts.utils import Registry


MODEL_REGISTRY = Registry(
    registry_name="model_registry",
    base_type=nn.Module,
    lazy_dirs=["dlts/models"],
)


def create_model(
        model_name: str,
        **kwargs
) -> nn.Module:
    """
    Create a model instance from the model registry.
    Args:
        model_name (str): The name of the model to create.
        **kwargs: Additional arguments to pass to the model constructor.
    Returns:
        nn.Module: An instance of the requested model.
    """
    create_fn = MODEL_REGISTRY.get(model_name)
    model = create_fn(**kwargs)
    logger.info(f'Creating model: {model_name}')

    return model

