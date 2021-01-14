import math
import os
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import SegmentationHead, initialization, Conv2dReLU
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet.decoder import CenterBlock
from torch import Tensor

from preprocess.dataset import get_training_validation_sets
from utils import set_deterministic_colab


class HookNetDecoderBlock(nn.Module):
    """
    The building block used by the U-Net decoder / expanding path to upsample the feature map resolution and
    downsample the number of channels.
    """

    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 use_batchnorm: bool = True):
        """

        :param in_channels: number of input channels
        :param skip_channels: number of channels added by the skip connections
        :param out_channels: number of output channels
        :param use_batchnorm: if True applies batch normalization after the 2 convolutional blocks
        """
        super().__init__()

        self.transposed_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self,
                x: Tensor,
                skip: Tensor = None) -> Tensor:
        """
        Computes the U-Net's expanding path.

        :param x: input image from the encoder
        :param skip: feature from the skip connections
        :return: the resulting feature map
        """
        # Upsample
        x = self.transposed_conv(x)
        # Concatenate features from the skip connection
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class HookNetDecoder(nn.Module):
    """
    The decoder / expanding branch used by HookNet. Its main difference with the classical one is that it returns the
    features at each depth of the decoder.
    """

    def __init__(
            self,
            encoder_channels: List[int],
            decoder_channels: List[int],
            n_blocks: int = 5,
            use_batchnorm: bool = True,
            center: bool = False):
        """

        :param encoder_channels: list containing for each depth of the encoding path the number channels
        :param decoder_channels: list containing for each depth of the decoding path the number of channels
        :param n_blocks: number of decoding blocks
        :param use_batchnorm: if True applies batch norm in the blocks
        :param center: if True crop the center of the feature map passed by the skip connection
        """

        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # Remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # Reverse channels to start from the head of the encoder
        encoder_channels = encoder_channels[::-1]

        # Computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)
        blocks = [
            HookNetDecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in
            zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features) -> Tuple[Tensor, List[Tensor]]:
        """

        :param features: the features from the encoder
        :return: the predicted mask and the features computed at each depth
        """

        # Remove first skip with same spatial resolution
        features = features[1:]
        # Reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        decoder_features = [x]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            decoder_features.append(x)

        return x, decoder_features


class HookNet(nn.Module):
    """
    A) Architecture

    Constraints:
    1) the two branches have the same architecture but do not share their weights
    2) each branch consists of an encoder-decoder model based on the U-Net


    B) Input patches

    The input to HookNet is a pair (P_c, P_t) of (M × M ×3) MFMR RGB concentric patches extracted at two different
    spatial resolutions res_c and res_t measured in µm/px for the context (C) and the target (T) branch,
    respectively. In this way, we ensure that the field of view of P_t corresponds to the central square region of
    size (M * res_t/res_c × M * res_t/res_c × 3).

    Constraints:
    1) M must be even
    2) 2^(D) * res_t >= res_c


    C) Hooking strategy

    "We propose to combine, i.e., hook-up information from the context branch into the target branch via the simple
    concatenation of feature maps extracted from the decoding paths of the two branches."

    The concatenation occurs at the beginning of the decoder of the target branch. There, the spatial resolution of the
    feature map is 2^d * r where d is the depth in the encoder-decoder model and r is the resolution of the input patch
    measured in µm/px.

    "To define the relative depths were the hooking can take place, we define a SFR ratio between a pair of feature
    maps as SRF_c / SRF_t = 2 ^ (d_c - d_t) * (res_c / res_t). Here d_t and d_c, are the relative depths for the target
    and context branch, respectively. In practice, hooking can take place when feature maps from both branches comprise
    the same resolution: SRF_c / SRF_t = 1."

    Example:
        res_c = 2.0 µm/px
        res_t = 0.5 µm/px
        d_t = 4 because "... hooking could be best done at the beginning of the decoder in the target branch, to take
        maximum advantage of the inherent up-sampling in the decoding path, where the concatenated feature maps can
        benefit from every skip connection within the target branch."

        So to solve the above equation we need to apply the hooking at d_c = 2.

        Context:                                        Target:

        2.0 µm/px depth 0                               0.5 µm/px depth 0
            4.0 µm/px depth 1                               1.0 µm/px depth 1
                8.0 µm/px depth 2                               2.0 µm/px depth 2
                    16.0 µm/px depth 3                              4.0 µm/px depth 3
                        32.0 µm/px depth 4          |------------>      8.0 µm/px depth 4
                    16.0 µm/px depth 3              |               4.0 µm/px depth 3
                8.0 µm/px depth 2   ----------------|           2.0 µm/px depth 2
            4.0 µm/px depth 1                               1.0 µm/px depth 1
        2.0 µm/px depth 0                                0.5 µm/px depth 0


    D) Loss function

    TODO

    """

    def __init__(self,
                 res_t,
                 res_c,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 verbose: bool = False):
        """
        :param res_t: resolution target in µm/px
        :param res_c: resolution context in µm/px
        :param encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
        to extract features of different spatial resolution
        :param encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
        two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
        with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
        Default is 5
        :param encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
        other pretrained weights (see table with available weights for each encoder_name)
        :param decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in
        decoder. Length of the list should be the same as **encoder_depth**
        :param decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
        is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
        Available options are **True, False, "inplace"**
        and **scse**. SCSE paper - https://arxiv.org/abs/1808.08127
        :param in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
        Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"identity"**, **callable** and **None**.
        Default is **None**
        :param verbose
        """
        super(HookNet, self).__init__()

        self.verbose = verbose

        # Check whether the different resolutions are legal and allow the hooking mechanism.
        assert 2 ** encoder_depth * res_t >= res_c, "2^(D) * res_t >= res_c must be true!"

        self.hook_depth = math.log2((res_t / res_c) * 2 ** encoder_depth)
        self.hook_filters = decoder_channels[int(self.hook_depth) - 1]
        print(f"Hooking occurs at depth {self.hook_depth} of the context decoder (filters = {self.hook_filters}).")

        # Context branch
        self.context_encoder = get_encoder(encoder_name,
                                           in_channels=in_channels,
                                           depth=encoder_depth,
                                           weights=encoder_weights)

        self.context_decoder = HookNetDecoder(encoder_channels=self.context_encoder.out_channels,
                                              decoder_channels=decoder_channels,
                                              n_blocks=encoder_depth,
                                              use_batchnorm=decoder_use_batchnorm,
                                              center=True if encoder_name.startswith("vgg") else False)

        self.context_head = SegmentationHead(in_channels=decoder_channels[-1],
                                             out_channels=classes,
                                             activation=activation,
                                             kernel_size=3)

        # Target branch
        self.target_encoder = get_encoder(encoder_name,
                                          in_channels=in_channels,
                                          depth=encoder_depth,
                                          weights=encoder_weights)

        self.target_head = SegmentationHead(in_channels=decoder_channels[-1],
                                            out_channels=classes,
                                            activation=activation,
                                            kernel_size=3)

        # Change the number of channels expected by the target branch decoder path because of hooking mechanism
        target_encoder_out_channels = list(self.target_encoder.out_channels)
        target_encoder_out_channels[-1] += decoder_channels[int(self.hook_depth) - 1]
        print("Encoder out channels with hooking", target_encoder_out_channels)

        self.target_decoder = HookNetDecoder(encoder_channels=target_encoder_out_channels,
                                             decoder_channels=decoder_channels,
                                             n_blocks=encoder_depth,
                                             use_batchnorm=decoder_use_batchnorm,
                                             center=True if encoder_name.startswith("vgg") else False)

        self.name = "h-{}".format(encoder_name)

        # Initialize decoder and heads of both context and target branches
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the layers of the model using different strategies depending on the decoder and segmentation heads.
        """

        initialization.initialize_decoder(self.target_decoder)
        initialization.initialize_decoder(self.context_decoder)

        initialization.initialize_head(self.target_head)
        initialization.initialize_head(self.context_head)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Sequentially pass the image trough model's encoder, decoder and heads of both target and context branches,
        combining their strategy with the the hooking strategy.

        :param x: the target and contextual input image
        :return: the predicted mask from the target and context branch
        """

        # Context branch
        # Use normally the context branch to extract the different features
        context_enc_features = self.context_encoder(x[1])
        # Get the features at each depth and the output
        context_output, context_dec_features = self.context_decoder(*context_enc_features)
        # Get the prediction of the context
        context_preds = self.context_head(context_output)

        # Target branch
        target_enc_features = self.target_encoder(x[0])

        # Hooking
        # Extract and cut from the specific depth the region and concatenate it to the target features in the bottleneck
        if self.verbose:
            print("Target encoding feature shape", target_enc_features[-1].shape)
            print("Context encoding feature shape", context_dec_features[int(self.hook_depth)].shape)

        # Get the head of the target's encoder
        target_head_width = target_enc_features[-1].shape[2]
        target_head_height = target_enc_features[-1].shape[3]

        context_hooked_width = context_dec_features[int(self.hook_depth)].shape[2]
        context_hooked_height = context_dec_features[int(self.hook_depth)].shape[3]

        hooked_feature = context_dec_features[int(self.hook_depth)][:,
                         :,
                         (context_hooked_width - target_head_width) // 2:
                         (context_hooked_width - target_head_width) // 2 + target_head_width,
                         (context_hooked_height - target_head_height) // 2:
                         (context_hooked_height - target_head_height) // 2 + target_head_height]

        if self.verbose:
            print("Hooked feature shape", hooked_feature.shape)

        # Hook
        target_enc_features[-1] = torch.cat([target_enc_features[-1], hooked_feature], dim=1)

        target_output, _ = self.target_decoder(*target_enc_features)
        # Get the prediction of the target
        target_preds = self.target_head(target_output)

        return target_preds, context_preds


def test_hooknet():
    images_path = os.path.join('data', '256x256', 'train')
    masks_path = os.path.join('data', '256x256', 'masks')

    seed = 42
    set_deterministic_colab(seed)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    training_dataset, _, _, _ = get_training_validation_sets(images_path,
                                                             masks_path,
                                                             0.3,
                                                             {'train': None, 'val': None},
                                                             'cpu',
                                                             mean=mean,
                                                             std=std)

    hooknet = HookNet(1.0, 2.0, 'efficientnet-b0')

    # print(hooknet)

    image1, mask1 = training_dataset[0]
    image2, mask2 = training_dataset[0]

    pred = hooknet((torch.stack([image1, image2]), torch.stack([image1, image2])))


if __name__ == "__main__":
    # execute only if run as a script
    test_hooknet()
