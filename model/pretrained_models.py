from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def get_deeplabv3_resnet(device: str,
                         pretrained: bool = True,
                         resnet_layers: int = 50,
                         freeze_backbone=True,
                         progress: bool = True):
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
    :return:
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
