import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50

__all__ = ["DeepLabV3_ResNet50"]


def DeepLabV3_ResNet50:
    weights = torch.load("deeplabv3_resnet50.pth")
    # weights are equivalent to
    # torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    return deeplabv3_resnet50( weights=weights )
