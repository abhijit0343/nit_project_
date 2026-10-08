import torch
import torch.nn as nn
import torchvision.models as models


class LiquidClassifier(nn.Module):
    """
    MobileNetV3-Small fine-tuned for liquid classification
    (e.g. mustard oil vs water).

    The ImageNet-pretrained backbone acts as a frozen feature extractor;
    only the lightweight custom head is trained by default, which keeps
    training fast even on small datasets.

    Args:
        num_classes     : number of output liquid classes (default 2)
        freeze_backbone : if True, backbone weights are frozen (default True)
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()

        # Load MobileNetV3-Small with ImageNet weights
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Optionally freeze the feature extractor
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # The default MobileNetV3-Small classifier expects 576 input features
        # Replace it with our custom binary/multi-class head
        backbone.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 3, 224, 224)
        return self.model(x)   # returns raw logits (batch, num_classes)
