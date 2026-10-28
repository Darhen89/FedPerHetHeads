import hydra
import numpy as np

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import sigmoid

from sklearn.metrics import multilabel_confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, f1_score

from torchvision.models.resnet import resnet34

import os
import json
import pandas as pd
from PIL import Image
from datasets import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, CenterCrop

from typing import Optional

from pathlib import Path

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = cfg
    device = cfg.server_device

    set_1_train_loader, set_1_test_loader = corel5k_preprocess("datasets/corel5k", cfg)

    trainloader = set_1_train_loader
    testloader = set_1_test_loader

    model = ResNet()
    model.to(device)

    pos_weight = torch.tensor((1, 4, 8, 10, 11, 15, 17, 19, 20, 21))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9
    )

    round_count = 0
    for _ in range(cfg.num_rounds):
        round_count = round_count + 1
        print("Round", round_count)
        correct, total = 0.0, 0.0
        loss: torch.Tensor = 0.0
        for _ in range(cfg.num_epochs):
            running_loss = 0.0
            for batch in trainloader:
                optimizer.zero_grad()
                outputs = model(batch["image_name"].to(device))
                labels = batch["image_labels"].to(device)
                loss = criterion(outputs, labels)

                preds = sigmoid(outputs)
                correct += label_ranking_average_precision_score(labels.tolist(), preds.tolist())

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                total += labels.size(0)

        print("loss: ", loss.item(), "accuracy: ", correct / total)

    #----------------EVALUATION--------------
    f1_s = np.zeros(10)
    total_tn = np.zeros(10)
    total_tp = np.zeros(10)
    total_fn = np.zeros(10)
    total_fp = np.zeros(10)

    #model.eval()
    with torch.no_grad():
        for batch in testloader:
            outputs = model(batch["image_name"].to(device))
            labels = batch["image_labels"].to(device)
            loss = criterion(outputs, labels)

            preds = sigmoid(outputs)
            preds_binary = (preds > 0.5).float()

            '''Referencias para las métricas:
            ranking de etiquetas:
            https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
            matriz de confusión y métricas derivadas
            https://medium.com/data-science/evaluating-multi-label-classifiers-a31be83da6ea
            https://medium.com/synthesio-engineering/precision-accuracy-and-f1-score-for-multi-label-classification-34ac6bdfb404
            https://scikit-learn.org/stable/modules/model_evaluation.html#multi-label-confusion-matrix
            '''

            # LABEL RANKING AVERAGE PRECISION: fracción de las etiquetas predichas que corresponden a las etiquetas reales
            #print("label ranking average precision: ", label_ranking_average_precision_score(labels.tolist(), preds.tolist()))
            # LABEL RANKING LOSS:
            #print("label ranking loss: ", label_ranking_loss(labels.tolist(), preds.tolist()))

            #F1 SCORE: media armónica de la precision y el recall
            #print("F1 score: ",f1_score(labels.tolist(), preds_binary.tolist(), average='micro'))

            #CONFUSION MATRIX
            confusion_matrix = multilabel_confusion_matrix(labels.tolist(), preds_binary.tolist())
            tn = confusion_matrix[:, 0, 0]
            tp = confusion_matrix[:, 1, 1]
            fn = confusion_matrix[:, 1, 0]
            fp = confusion_matrix[:, 0, 1]

            f1_s += f1_score(labels.tolist(), preds_binary.tolist(), average=None, zero_division=0.0)
            total_tn += tn
            total_tp += tp
            total_fn += fn
            total_fp += fp

            #print("accuracy: ", (tp + tn) / (tp + tn + fp + fn)) #proporcion de ejemplos correctamente clasificados
            #print("precision: ",tp / (tp + fp)) #proporción de predicciones correctas entre el total de predicciones para una clase
            #print("recall: ",tp / (tp + fn)) #proporción de veces que una clase ha sido predicha correctamente, probabilidad de positivos verdaderos
            #print("specificity",tn / (tn + fp)) #probabilidad de negativos verdaderos
            #print("fall out: ",fp / (fp + tn)) #probabilidad de falsos positivos
            #print("miss rate: ",fn / (fn + tp)) #probabilidad de falsos negativos

        total = len(testloader)

        f1_s = f1_s / total
        accuracy = ((total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)) / total #proporcion de ejemplos correctamente clasificados
        precision = (total_tp / (total_tp + total_fp)) / total #proporción de predicciones correctas entre el total de predicciones para una clase
        recall = (total_tp / (total_tp + total_fn)) / total #proporción de veces que una clase ha sido predicha correctamente, probabilidad de positivos verdaderos
        specificity = (total_tn / (total_tn + total_fp)) / total #probabilidad de negativos verdaderos
        fall_out = (total_fp / (total_fp + total_tn)) / total #probabilidad de falsos positivos
        miss_rate = (total_fn / (total_fn + total_tp)) / total #probabilidad de falsos negativos

        report = {'water': {'f1_score': round(f1_s[0],4),
                            'accuracy': round(accuracy[0],4),
                            'precision': round(precision[0],4),
                            'recall': round(recall[0],4),
                            'specificity': round(specificity[0],4),
                            'fall_out': round(fall_out[0],4),
                            'miss_rate': round(miss_rate[0],4),},
                'buildings': {'f1_score': round(f1_s[1],4),
                            'accuracy': round(accuracy[1],4),
                            'precision': round(precision[1],4),
                            'recall': round(recall[1],4),
                            'specificity': round(specificity[1],4),
                            'fall_out': round(fall_out[1],4),
                            'miss_rate': round(miss_rate[1],4),},
                'rocks': {'f1_score': round(f1_s[2],4),
                            'accuracy': round(accuracy[2],4),
                            'precision': round(precision[2],4),
                            'recall': round(recall[2],4),
                            'specificity': round(specificity[2],4),
                            'fall_out': round(fall_out[2],4),
                            'miss_rate': round(miss_rate[2],4),},
                'field': {'f1_score': round(f1_s[3],4),
                        'accuracy': round(accuracy[3],4),
                        'precision': round(precision[3],4),
                        'recall': round(recall[3],4),
                        'specificity': round(specificity[3],4),
                        'fall_out': round(fall_out[3],4),
                        'miss_rate': round(miss_rate[3],4), },
                'sand': {'f1_score': round(f1_s[4],4),
                        'accuracy': round(accuracy[4],4),
                        'precision': round(precision[4],4),
                        'recall': round(recall[4],4),
                        'specificity': round(specificity[4],4),
                        'fall_out': round(fall_out[4],4),
                        'miss_rate': round(miss_rate[4],4), },
                'cars': {'f1_score': round(f1_s[5],4),
                        'accuracy': round(accuracy[5],4),
                        'precision': round(precision[5],4),
                        'recall': round(recall[5],4),
                        'specificity': round(specificity[5],4),
                        'fall_out': round(fall_out[5],4),
                        'miss_rate': round(miss_rate[5],4), },
                'polar': {'f1_score': round(f1_s[6],4),
                        'accuracy': round(accuracy[6],4),
                        'precision': round(precision[6],4),
                        'recall': round(recall[6],4),
                        'specificity': round(specificity[6],4),
                        'fall_out': round(fall_out[6],4),
                        'miss_rate': round(miss_rate[6],4), },
                'ruins': {'f1_score': round(f1_s[7],4),
                        'accuracy': round(accuracy[7],4),
                        'precision': round(precision[7],4),
                        'recall': round(recall[7],4),
                        'specificity': round(specificity[7],4),
                        'fall_out': round(fall_out[7],4),
                        'miss_rate': round(miss_rate[7],4), },
                'statue': {'f1_score': round(f1_s[8],4),
                        'accuracy': round(accuracy[8],4),
                        'precision': round(precision[8],4),
                        'recall': round(recall[8],4),
                        'specificity': round(specificity[8],4),
                        'fall_out': round(fall_out[8],4),
                        'miss_rate': round(miss_rate[8],4), },
                'wall': {'f1_score': round(f1_s[9],4),
                        'accuracy': round(accuracy[9],4),
                        'precision': round(precision[9],4),
                        'recall': round(recall[9],4),
                        'specificity': round(specificity[9],4),
                        'fall_out': round(fall_out[9],4),
                        'miss_rate': round(miss_rate[9],4), },
                }
        print("Classification report")
        print("Class      f1_score accuracy precision recall specificity fall_out miss_rate")
        for stud, val in report.items():
            print("{:10} {:^8} {:^8} {:^9} {:^6} {:^11} {:^8} {:^9}".format(stud, val["f1_score"], val["accuracy"], val["precision"], val["recall"], val["specificity"],
                                                              val["fall_out"], val["miss_rate"]))




def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

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
            nn.Linear(512, 10),
        )
        self.body = nn.Sequential(*list(self.body.children())[:-2])
        body_layer4 = list(self.body.children())[-1]
        self.body = nn.Sequential(*list(self.body.children())[:-1])
        self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward inputs through the model."""
        x = self.body(x)
        return self.head(x)



def corel5k_preprocess(root, config):
    print("Preprocessing Corel-5k dataset...")

    # define dataset path
    train_file_path = Path(root, "train.json")
    train_aug_file_path = Path(root, "train_aug.json")
    test_file_path = Path(root, "test.json")
    images_path = Path(root, "images")

    # read the files to create pandas dataframe
    with open(train_file_path) as json_data:
        train_data = json.load(json_data)
        train_df = pd.DataFrame(train_data['samples'])
        train_df['image_name'] = train_df['image_name'].apply(
            lambda image_name: Image.open(os.path.join(images_path, image_name)).convert('RGB'))

    with open(train_aug_file_path) as json_data:
        train_aug_data = json.load(json_data)
        train_aug_df = pd.DataFrame(train_data['samples'])
        train_aug_df['image_name'] = train_aug_df['image_name'].apply(
            lambda image_name: Image.open(os.path.join(images_path, image_name)).convert('RGB'))

    with open(test_file_path) as json_data:
        test_data = json.load(json_data)
        test_df = pd.DataFrame(test_data['samples'])
        test_df['image_name'] = test_df['image_name'].apply(
            lambda image_name: Image.open(os.path.join(images_path, image_name)).convert('RGB'))

    joined_df = pd.concat([train_df, train_aug_df, test_df])

    set_1_labels = ['water', 'buildings', 'rocks', 'field', 'sand', 'cars', 'polar', 'ruins', 'statue', 'wall']

    set_1_df = joined_df.loc[
        joined_df['image_labels'].apply(lambda labels: any(label in labels for label in set_1_labels))]
    set_1_df.loc[:, 'image_labels'] = set_1_df['image_labels'].apply(
        lambda labels: [label for label in labels if label in set_1_labels])
    set_1_df.reset_index(drop=True, inplace=True)

    set_1_ds = Dataset.from_dict(set_1_df.to_dict(orient='list'))


    set_1_ds_partition = set_1_ds.train_test_split(test_size=0.2)


    train_set_1_ds = set_1_ds_partition['train']
    test_set_1_ds = set_1_ds_partition['test']

    set_1_map = {'water': 0, 'buildings': 1, 'rocks': 2, 'field': 3, 'sand': 4, 'cars': 5, 'polar': 6, 'ruins': 7, 'statue': 8, 'wall': 9}

    def remap_set_1_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(set_1_map[label])
        example["image_labels"] = mapped_labels
        return example

    remapped_train_set_1_ds = train_set_1_ds.map(remap_set_1_labels)
    remapped_test_set_1_ds = test_set_1_ds.map(remap_set_1_labels)

    pytorch_transforms = Compose([ToTensor(),
                                  Resize(size=(32, 32)),
                                  RandomHorizontalFlip(),
                                  RandomRotation(30),
                                  CenterCrop(size=(32, 32)),
                                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        if isinstance(batch['image_name'], list):
            batch['image_name'] = torch.stack([pytorch_transforms(img) for img in batch['image_name']])
        else:
            batch['image_name'] = pytorch_transforms(batch['image'])
        # One-hot encode the multilabels
        all_labels = [torch.tensor(labels) for labels in batch['image_labels']]
        # Create one-hot encoding for each image's classes
        one_hot_labels = [nn.functional.one_hot(label, num_classes=10).sum(dim=0) for label in all_labels]
        # Stack them into a batch
        batch['image_labels'] = torch.stack(one_hot_labels)

        return batch

    remapped_train_set_1_ds = remapped_train_set_1_ds.with_transform(apply_transforms)
    remapped_test_set_1_ds = remapped_test_set_1_ds.with_transform(apply_transforms)

    def collate_fn(batch):
        return {
            'image_name': torch.stack([x['image_name'] for x in batch]),
            'image_labels': torch.stack([x['image_labels'] for x in batch]).float()
        }

    set_1_train_loader = DataLoader(remapped_train_set_1_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)
    set_1_test_loader = DataLoader(remapped_test_set_1_ds, batch_size=64, drop_last=True, collate_fn=collate_fn)
    return set_1_train_loader, set_1_test_loader

main()