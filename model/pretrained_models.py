from typing import Tuple, List, Dict

import torch.nn
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import segmentation_models_pytorch as smp


def get_deeplabv3_resnet(device: str,
                         pretrained: bool = True,
                         resnet_layers: int = 50,
                         freeze_backbone: bool = True,
                         progress: bool = True) -> torch.nn.Module:
    """
    Constructs a DeepLabV3 model with a ResNet-50/101 backbone.
    https://arxiv.org/abs/1706.05587

    The pre-trained models expect input images normalized in the same way. The images have to be loaded in to a range
    of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. They have been
    trained on images resized such that their minimum size is 520.

    The pre-trained models have been trained on a subset of COCO train2017, on the 20 categories that are present in
    the Pascal VOC dataset.
    https://pytorch.org/docs/stable/torchvision/models.html#deeplabv3

    :param device: device where the model is loaded
    :param pretrained: if True the model is loaded with pretrained weights (COCO train2017)
    :param resnet_layers: type of backbone, 101 = ResNet-101, 50 = ResNet-50
    :param freeze_backbone: if True backbone's parameters are set to not require grads
    :param progress: if True a progress bar is printed
    :return: the model
    """

    if resnet_layers == 101:
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=progress).to(device)
    elif resnet_layers == 50:
        model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=progress).to(device)

    else:
        raise ValueError('resNetLayer can assume only values 101 and 50.')

    model.classifier = DeepLabHead(2048, 1)
    model.backbone.requires_grad = not freeze_backbone
    return model


def get_unet(device: str,
             encoder_weights: str = 'imagenet',
             encoder: str = 'efficientnet-b0',
             decoder_channels: List[int] = (256, 128, 64, 32, 16),
             freeze_backbone: bool = True,
             depth: int = 5,
             activation: str = 'sigmoid') -> Tuple[torch.nn.Module, Dict]:
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

    # TODO Freeze backbone

    return model, smp.encoders.get_preprocessing_params(encoder)
