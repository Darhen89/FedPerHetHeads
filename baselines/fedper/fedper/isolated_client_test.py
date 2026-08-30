import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34

from typing import Optional

from fedper.dataset_preparation import dataset_class_type_split
from fedper.utils import (
    set_num_classes,
    )

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding."""

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    """Basic block for ResNet."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward inputs through the block."""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(ResNet, self).__init__()
        self.body = resnet34()
        self.head = nn.Sequential(
            BasicBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 6),
        )
        self.body = nn.Sequential(*list(self.body.children())[:-2])
        body_layer4 = list(self.body.children())[-1]
        self.body = nn.Sequential(*list(self.body.children())[:-1])
        self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward inputs through the model."""
        x = self.body(x)
        return self.head(x)

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = cfg = set_num_classes(cfg)
    device = cfg.server_device

    animalstrainloader, animalstestloader, vehiclestrainloader, vehiclestestloader = dataset_class_type_split(cfg, 0)
    trainloader = animalstrainloader
    testloader = animalstestloader
    model = ResNet()
    model.to(device)
    #model.train()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.learning_rate, momentum=0.9
    )
    correct, total = 0, 0
    loss: torch.Tensor = 0.0
    round_count = 0
    for _ in range(cfg.num_rounds):
        round_count = round_count+1
        print("ROUND: ", round_count)
        for _ in range(cfg.num_epochs):
            for batch in trainloader:
                optimizer.zero_grad()
                outputs = model(batch["img"].to(device))
                labels = batch["label"].to(device)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        print("loss: ", loss.item(), "accuracy: ", correct / total)

        model.to(device)
        correct, total, loss = 0, 0, 0.0
        # self.model.eval()
        with torch.no_grad():
            for batch in testloader:
                outputs = model(batch["img"].to(device))
                labels = batch["label"].to(device)
                loss += criterion(outputs, labels).item()
                total += batch["label"].size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        print("Test Accuracy: {:.4f}".format(correct / total))
        print("loss: ", loss / len(testloader.dataset), "accuracy: ", correct / total)

main()