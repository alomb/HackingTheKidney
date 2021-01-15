from torch import nn
from typing import Union


def freeze_backbone_layers(backbone: nn.Module, freeze: Union[bool, int]):
    if type(freeze) is bool:
        for layer in backbone.modules():
            set_requires_grad_for_layer(layer, not freeze)
    else:
        for layer in list(backbone.modules())[:-freeze]:
            set_requires_grad_for_layer(layer, False)


def set_requires_grad_for_layer(layer: nn.Module, train: bool) -> None:
    """
    Sets the attribute requires_grad to True or False for each parameter.

    :param layer: the layer to freeze.
    :param train: if true train the layer.
    """
    for p in layer.parameters():
        p.requires_grad = train
