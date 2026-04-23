import torch
import torchvision
# from torchinfo import summary

class DeepLabV3(torch.nn.Module):
    def __init__(self, nbClassses: int, backbone = "Resnet50")-> None:
        super().__init__()
        if backbone=="Resnet50":
            self.model=torchvision.models.segmentation.deeplabv3_resnet50(
                weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights,
                backbone=torchvision.models.ResNet50_Weights
            )
        elif backbone == "Resnet101":
            self.model=torchvision.models.segmentation.deeplabv3_resnet101(
                weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights,
                backbone=torchvision.models.ResNet101_Weights
            )
            for param in self.model.parameters():
                param.requires_grad=False
                
        self.model.classifier= torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, nbClassses+1)
        self.model.aux_classifier=None


    def forward(self, x: torch.tensor):
        return self.model(x)
    