"""ResNet model, model manager and split."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from omegaconf import DictConfig
from torch import sigmoid
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34
from sklearn.metrics import f1_score, label_ranking_average_precision_score, multilabel_confusion_matrix

from fedper.models import ModelManager, ModelSplit


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
    """ResNet model."""

    def __init__(
        self,
        num_head_layers: int = 1,
        num_classes: int = 10,
    ) -> None:
        super(ResNet, self).__init__()

        assert (
            num_head_layers > 0 and num_head_layers <= 17
        ), "num_head_layers must be greater than 0 and less than 16"

        self.num_head_layers = num_head_layers
        self.body = resnet34()

        # if only one head layer
        if self.num_head_layers == 1:
            self.head = self.body.fc
            self.body.fc = nn.Identity()
        elif self.num_head_layers == 2:
            self.head = nn.Sequential(
                BasicBlock(512, 512),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
            # remove head layers from body
            self.body = nn.Sequential(*list(self.body.children())[:-2])
            body_layer4 = list(self.body.children())[-1]
            self.body = nn.Sequential(*list(self.body.children())[:-1])
            self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-1])
        elif self.num_head_layers == 3:
            self.head = nn.Sequential(
                BasicBlock(512, 512),
                BasicBlock(512, 512),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
            # remove head layers from body
            self.body = nn.Sequential(*list(self.body.children())[:-2])
            body_layer4 = list(self.body.children())[-1]
            self.body = nn.Sequential(*list(self.body.children())[:-1])
            self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-2])
        else:
            raise NotImplementedError("Only 1 or 2 head layers supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward inputs through the model."""
        print("Forwarding through ResNet model")
        x = self.body(x)
        return self.head(x)


class ResNetModelSplit(ModelSplit):
    """Split ResNet model into body and head."""

    def _get_model_parts(self, model: ResNet) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


class ResNetModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_save_path: Optional[str],
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        learning_rate: float = 0.01,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_save_path: Path to save the client state.
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            trainloader: DataLoader containing the train data.
            testloader: DataLoader containing the test data.
            learning_rate: Learning rate for the optimizer.
        """

        super().__init__(
            model_split_class=ResNetModelSplit,
            client_id=client_id,
            config=config,
        )

        self.client_save_path = client_save_path
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config["server_device"]
        self.learning_rate = learning_rate


    def _create_model(self) -> nn.Module:
        """Return MobileNet-v1 model to be splitted into head and body."""
        try:
            return ResNet(
                num_head_layers=self.config["model"]["num_head_layers"],
                num_classes=self.config["model"]["num_classes"],
            ).to(self.device)
        except AttributeError:
            self.device = self.config["server_device"]
            return ResNet(
                num_head_layers=self.config["model"]["num_head_layers"],
                num_classes=self.config["model"]["num_classes"],
            ).to(self.device)

    def train(
        self,
        epochs: int = 1,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Method adapted from simple MobileNet-v1 (PyTorch) \
        https://github.com/wjc852456/pytorch-mobilenet-v1.

        Args:
            epochs: number of training epochs.

        Returns
        -------
            Dict containing the train metrics.
        """
        # Load client state (head) if client_save_path is not None and it is not empty
        if self.client_save_path is not None:
            try:
                self.model.head.load_state_dict(torch.load(self.client_save_path))
            except FileNotFoundError:
                print("No client state found, training from scratch.")
                pass

        pos_weigh_set_1 = torch.tensor((1,4,8,10,11,15,17,19,20,21))
        pos_weigh_set_2 = torch.tensor((1,5,8,9,10,14,16,16,18,20))
        pos_weigh_set_3 = torch.tensor((1,6,6,9,10,14,15,16,18,20))
        pos_weigh_set_4 = torch.tensor((2,3,7,9,12,12,15,17,19,19))

        pos_weight = np.zeros(10)

        if self.client_id == 0:
            pos_weight = pos_weigh_set_1
        elif self.client_id == 1:
            pos_weight = pos_weigh_set_2
        elif self.client_id == 2:
            pos_weight = pos_weigh_set_3
        elif self.client_id == 3:
            pos_weight = pos_weigh_set_4

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9
        )

        correct, total = 0.0, 0.0
        loss: torch.Tensor = 0.0
        for _ in range(epochs):
            running_loss = 0.0
            for batch in self.trainloader:
                optimizer.zero_grad()
                outputs = self.model(batch["image_name"].to(self.device))
                labels = batch["image_labels"].to(self.device)
                loss = criterion(outputs, labels)

                preds = sigmoid(outputs)
                correct += label_ranking_average_precision_score(labels.tolist(), preds.tolist())

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                total += labels.size(0)

        # Save client state (head)
        if self.client_save_path is not None:
            torch.save(self.model.head.state_dict(), self.client_save_path)

        return {"loss": loss.item(), "accuracy": correct / total}

    def test(
        self,
    ) -> Dict[str, float]:
        """Test the model maintained in self.model."""
        # Load client state (head)
        if self.client_save_path is not None:
            self.model.head.load_state_dict(torch.load(self.client_save_path))

        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        f1_s = 0.0
        f1_s_by_class = np.zeros(10)
        total_tn = np.zeros(10)
        total_tp = np.zeros(10)
        total_fn = np.zeros(10)
        total_fp = np.zeros(10)

        loss: torch.Tensor = 0.0

        correct = 0.0

        # self.model.eval()
        with torch.no_grad():
            for batch in self.testloader:
                outputs = self.model(batch["image_name"].to(self.device))
                labels = batch["image_labels"].to(self.device)
                loss = criterion(outputs, labels)

                preds = sigmoid(outputs)
                preds_binary = (preds > 0.5).float()

                # CONFUSION MATRIX
                confusion_matrix = multilabel_confusion_matrix(labels.tolist(), preds_binary.tolist())
                tn = confusion_matrix[:, 0, 0]
                tp = confusion_matrix[:, 1, 1]
                fn = confusion_matrix[:, 1, 0]
                fp = confusion_matrix[:, 0, 1]

                f1_s += f1_score(labels.tolist(), preds_binary.tolist(), average='micro', zero_division=0.0)
                f1_s_by_class += f1_score(labels.tolist(), preds_binary.tolist(), average=None, zero_division=0.0)

                total_tn += tn
                total_tp += tp
                total_fn += fn
                total_fp += fp

                correct += label_ranking_average_precision_score(labels.tolist(), preds.tolist())

        if self.client_save_path is not None:
            torch.save(self.model.head.state_dict(), self.client_save_path)

        total = len(self.testloader)

        f1_s = f1_s / total
        f1_s_by_class = f1_s_by_class / total
        accuracy = ((total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)) / total  # proporcion de ejemplos correctamente clasificados
        precision = (total_tp / (total_tp + total_fp)) / total  # proporción de predicciones correctas entre el total de predicciones para una clase
        recall = (total_tp / (total_tp + total_fn)) / total  # proporción de veces que una clase ha sido predicha correctamente, probabilidad de positivos verdaderos
        specificity = (total_tn / (total_tn + total_fp)) / total  # probabilidad de negativos verdaderos
        fall_out = (total_fp / (total_fp + total_tn)) / total  # probabilidad de falsos positivos
        miss_rate = (total_fn / (total_fn + total_tp)) / total  # probabilidad de falsos negativos

        report = self.create_report(self.client_id,f1_s_by_class, accuracy, precision, recall, specificity, fall_out, miss_rate)

        print("Classification report")
        print("Class      f1_score accuracy precision recall specificity fall_out miss_rate")
        for stud, val in report.items():
            print("{:10} {:^8} {:^8} {:^9} {:^6} {:^11} {:^8} {:^9}".format(stud, val["f1_score"], val["accuracy"], val["precision"], val["recall"], val["specificity"],
                                                              val["fall_out"], val["miss_rate"]))

        return {
            "loss": loss.item(),
            "f1_s": f1_s,
            #"f1_s_by_class": f1_s_by_class.tolist(),
            #"accuracy": accuracy.tolist(),
            #"precision": precision.tolist(),
            #"recall": recall.tolist(),
            #"specificity": specificity.tolist(),
            #"fall_out": fall_out.tolist(),
            #"miss_rate": miss_rate.tolist(),
        }

    def create_report(self,
                      client_id,
                      f1_s: ndarray,
                      accuracy: ndarray,
                      precision: ndarray,
                      recall: ndarray,
                      specificity: ndarray,
                      fall_out: ndarray,
                      miss_rate: ndarray) -> dict:

        set_labels = []

        set_1_labels = ['water', 'buildings', 'rocks', 'field', 'sand', 'cars', 'polar', 'ruins', 'statue', 'wall']
        set_2_labels = ['sky', 'mountain', 'stone', 'street', 'birds', 'leaf', 'valley', 'garden', 'horses', 'cat']
        set_3_labels = ['tree', 'flowers', 'snow', 'plane', 'beach', 'plants', 'house', 'hills', 'tracks', 'ocean']
        set_4_labels = ['people', 'grass', 'clouds', 'bear', 'boats', 'jet', 'bridge', 'close-up', 'sun', 'ice']

        if self.client_id == 0:
            set_labels = set_1_labels
        elif self.client_id == 1:
            set_labels = set_2_labels
        elif self.client_id == 2:
            set_labels = set_3_labels
        elif self.client_id == 3:
            set_labels = set_4_labels

        report = {set_labels[0]: {'f1_score': round(f1_s[0], 4),
                            'accuracy': round(accuracy[0], 4),
                            'precision': round(precision[0], 4),
                            'recall': round(recall[0], 4),
                            'specificity': round(specificity[0], 4),
                            'fall_out': round(fall_out[0], 4),
                            'miss_rate': round(miss_rate[0], 4), },
                  set_labels[1]: {'f1_score': round(f1_s[1], 4),
                                'accuracy': round(accuracy[1], 4),
                                'precision': round(precision[1], 4),
                                'recall': round(recall[1], 4),
                                'specificity': round(specificity[1], 4),
                                'fall_out': round(fall_out[1], 4),
                                'miss_rate': round(miss_rate[1], 4), },
                  set_labels[2]: {'f1_score': round(f1_s[2], 4),
                            'accuracy': round(accuracy[2], 4),
                            'precision': round(precision[2], 4),
                            'recall': round(recall[2], 4),
                            'specificity': round(specificity[2], 4),
                            'fall_out': round(fall_out[2], 4),
                            'miss_rate': round(miss_rate[2], 4), },
                  set_labels[3]: {'f1_score': round(f1_s[3], 4),
                            'accuracy': round(accuracy[3], 4),
                            'precision': round(precision[3], 4),
                            'recall': round(recall[3], 4),
                            'specificity': round(specificity[3], 4),
                            'fall_out': round(fall_out[3], 4),
                            'miss_rate': round(miss_rate[3], 4), },
                  set_labels[4]: {'f1_score': round(f1_s[4], 4),
                           'accuracy': round(accuracy[4], 4),
                           'precision': round(precision[4], 4),
                           'recall': round(recall[4], 4),
                           'specificity': round(specificity[4], 4),
                           'fall_out': round(fall_out[4], 4),
                           'miss_rate': round(miss_rate[4], 4), },
                  set_labels[5]: {'f1_score': round(f1_s[5], 4),
                           'accuracy': round(accuracy[5], 4),
                           'precision': round(precision[5], 4),
                           'recall': round(recall[5], 4),
                           'specificity': round(specificity[5], 4),
                           'fall_out': round(fall_out[5], 4),
                           'miss_rate': round(miss_rate[5], 4), },
                  set_labels[6]: {'f1_score': round(f1_s[6], 4),
                            'accuracy': round(accuracy[6], 4),
                            'precision': round(precision[6], 4),
                            'recall': round(recall[6], 4),
                            'specificity': round(specificity[6], 4),
                            'fall_out': round(fall_out[6], 4),
                            'miss_rate': round(miss_rate[6], 4), },
                  set_labels[7]: {'f1_score': round(f1_s[7], 4),
                            'accuracy': round(accuracy[7], 4),
                            'precision': round(precision[7], 4),
                            'recall': round(recall[7], 4),
                            'specificity': round(specificity[7], 4),
                            'fall_out': round(fall_out[7], 4),
                            'miss_rate': round(miss_rate[7], 4), },
                  set_labels[8]: {'f1_score': round(f1_s[8], 4),
                             'accuracy': round(accuracy[8], 4),
                             'precision': round(precision[8], 4),
                             'recall': round(recall[8], 4),
                             'specificity': round(specificity[8], 4),
                             'fall_out': round(fall_out[8], 4),
                             'miss_rate': round(miss_rate[8], 4), },
                  set_labels[9]: {'f1_score': round(f1_s[9], 4),
                           'accuracy': round(accuracy[9], 4),
                           'precision': round(precision[9], 4),
                           'recall': round(recall[9], 4),
                           'specificity': round(specificity[9], 4),
                           'fall_out': round(fall_out[9], 4),
                           'miss_rate': round(miss_rate[9], 4), },
                  }

        return report

    def train_dataset_size(self) -> int:
        """Return train data set size."""
        return len(self.trainloader)

    def test_dataset_size(self) -> int:
        """Return test data set size."""
        return len(self.testloader)

    def total_dataset_size(self) -> int:
        """Return total data set size."""
        return len(self.trainloader) + len(self.testloader)
