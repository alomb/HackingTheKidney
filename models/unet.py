from torch import nn
import segmentation_models_pytorch as smp
from .utils import freeze_backbone_layers
from typing import List, Tuple, Dict, Union


def get_unet(device: str,
             encoder_weights: str = 'imagenet',
             encoder: str = 'efficientnet-b0',
             decoder_channels: List[int] = (256, 128, 64, 32, 16),
             freeze_backbone: Union[bool, int] = True,
             depth: int = 5,
             activation: str = None) -> Tuple[nn.Module, Dict]:
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
    :param freeze_backbone: if True backbone's parameters are set to not require grads.
    If an int is passed, all the layers but the last n ones are freezed
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

    freeze_backbone_layers(model.encoder, freeze_backbone)

    return model, smp.encoders.get_preprocessing_params(encoder)
