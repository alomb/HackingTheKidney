from typing import Tuple, List, Dict

import torch.nn

import segmentation_models_pytorch as smp


def set_requires_grad_for_layer(layer: torch.nn.Module, train: bool) -> None:
    """Sets the attribute requires_grad to True or False for each parameter.

        Args:
            layer: the layer to freeze.
            train: if true train the layer.
    """
    for p in layer.parameters():
        p.requires_grad = train


def get_unet(device: str,
             encoder_weights: str = 'imagenet',
             encoder: str = 'efficientnet-b0',
             decoder_channels: List[int] = (256, 128, 64, 32, 16),
             freeze_backbone: bool = True,
             depth: int = 5,
             activation: str = None) -> Tuple[torch.nn.Module, Dict]:
    """
    Constructs a Unet model with a custom backbone.
    https://arxiv.org/abs/1505.04597

    Information regarding the possible encoders and their weights are available in the following link
    https://github.com/qubvel/segmentation_models.pytorch#encoders-

    Information (channels, range of values, mean, std) regarding the preprocessing step on the pretrained encoder is
    returned as well.

    :param device: device where the model is loaded
    :param encoder: type of encoder
    :param encoder_weights: the dataset where the encoder was pretrained
    :param decoder_channels: integers which specify in_channels parameter for convolutions used in decoder.
    Its length should be the same as **encoder_depth**
    :param freeze_backbone: if True backbone's parameters are set to not require grads
    :param depth: specify a number of downsampling operations in the encoder. You can make your model lighter
    specifying smaller depth.
    :param activation: An activation function to apply after the final convolution layer. Available options are
    "sigmoid", "softmax", "logsoftmax", "identity", callable and None. Default is None.
    :return: the model abd the preprocessing parameters associated to the pretrained model
    """

    model = smp.Unet(encoder_name=encoder,
                     encoder_depth=depth,
                     encoder_weights=encoder_weights,
                     decoder_channels=decoder_channels,
                     decoder_attention_type=None,
                     in_channels=3,
                     classes=1,
                     activation=activation).to(device)

    for layer in model.encoder.modules():
        set_requires_grad_for_layer(layer, freeze_backbone)

    return model, smp.encoders.get_preprocessing_params(encoder)


def get_deeplabv3(
    encoder_weights: str = 'imagenet',
    encoder: str = 'resnet34',
    decoder_channels: int = 256,
    freeze_backbone: bool = True,
    depth: int = 5,
    activation: str = None,
    device: str = None
) -> Tuple[torch.nn.Module, Dict]:
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
                          activation=activation)

    if device is not None:
        model = model.to(device)

    for layer in model.encoder.modules():
        set_requires_grad_for_layer(layer, freeze_backbone)

    return model, smp.encoders.get_preprocessing_params(encoder)
