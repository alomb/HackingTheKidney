from typing import Tuple, Union, Dict
import torch.nn
import segmentation_models_pytorch as smp
from .utils import freeze_backbone_layers


def get_deeplabv3(device: str,
                  encoder_weights: str = 'imagenet',
                  encoder: str = 'resnet50',
                  decoder_channels: int = 256,
                  freeze_backbone: Union[bool, int] = True,
                  depth: int = 5,
                  activation: str = None) -> Tuple[torch.nn.Module, Dict]:
    """
    Constructs a DeepLabV3 model with a custom backbone.
    https://arxiv.org/abs/1706.05587

    Information regarding the possible encoders and their weights are available in the following link
    https://github.com/qubvel/segmentation_models.pytorch#encoders-

    Information (channels, range of values, mean, std) regarding the preprocessing step on the pretrained encoder is
    returned as well.

    :param device: device where the model is loaded
    :param encoder: type of encoder
    :param encoder_weights: the dataset where the encoder was pretrained
    :param decoder_channels: a number of convolution filters in ASPP module. Default is 256
    :param freeze_backbone: if True backbone's parameters are set to not require grads
    :param depth: a number of stages used in encoder in range [3, 5]. Each stage generate features 2 times smaller in
    spatial dimensions than previous one (e.g. for depth 0 we will have features with shapes [(N, C, H, W),],
    for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
    :param activation: An activation function to apply after the final convolution layer. Available options are
    "sigmoid", "softmax", "logsoftmax", "identity", callable and None. Default is None.
    :return: the model abd the preprocessing parameters associated to the pretrained model
    """

    model = smp.DeepLabV3(encoder_name=encoder,
                          encoder_depth=depth,
                          encoder_weights=encoder_weights,
                          decoder_channels=decoder_channels,
                          upsampling=8,
                          in_channels=3,
                          classes=1,
                          activation=activation).to(device)

    freeze_backbone_layers(model.encoder, freeze_backbone)

    return model, smp.encoders.get_preprocessing_params(encoder)


def get_deeplabv3plus(device: str,
                      encoder_weights: str = 'imagenet',
                      encoder: str = 'resnet50',
                      decoder_channels: int = 256,
                      freeze_backbone: Union[bool, int] = True,
                      depth: int = 5,
                      activation: str = None) -> Tuple[torch.nn.Module, Dict]:
    """
    Constructs a DeepLabV3+ model with a custom backbone.
    https://arxiv.org/abs/1802.02611

    Information regarding the possible encoders and their weights are available in the following link
    https://github.com/qubvel/segmentation_models.pytorch#encoders-

    Information (channels, range of values, mean, std) regarding the preprocessing step on the pretrained encoder is
    returned as well.

    :param device: device where the model is loaded
    :param encoder: type of encoder
    :param encoder_weights: the dataset where the encoder was pretrained
    :param decoder_channels: a number of convolution filters in ASPP module. Default is 256
    :param freeze_backbone: if True backbone's parameters are set to not require grads
    :param depth: a number of stages used in encoder in range [3, 5]. Each stage generate features 2 times smaller in
    spatial dimensions than previous one (e.g. for depth 0 we will have features with shapes [(N, C, H, W),],
    for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). Default is 5
    :param activation: An activation function to apply after the final convolution layer. Available options are
    "sigmoid", "softmax", "logsoftmax", "identity", callable and None. Default is None.
    :return: the model abd the preprocessing parameters associated to the pretrained model
    """

    model = smp.DeepLabV3Plus(encoder_name=encoder,
                              encoder_depth=depth,
                              encoder_weights=encoder_weights,
                              decoder_channels=decoder_channels,
                              upsampling=8,
                              in_channels=3,
                              classes=1,
                              activation=activation).to(device)

    freeze_backbone_layers(model.encoder, freeze_backbone)

    return model, smp.encoders.get_preprocessing_params(encoder)
