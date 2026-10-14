import hydra
import os
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import json
from PIL import Image

from datasets import Dataset

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, \
    CenterCrop

from omegaconf import DictConfig
import torch
import torch.nn as nn




@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    print("Preprocessing Corel-5k dataset...")

    root = "datasets/corel5k/"

    # get number of clients
    num_clients = cfg["num_clients"]

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

    set_1_labels = ['water','buildings','rocks','field','sand','cars','polar','ruins','statue','wall']
    set_2_labels = ['sky','mountain','stone','street','birds','leaf','valley','garden','horses','cat']
    set_3_labels = ['tree','flowers','snow','plane','beach','plants','house','hills','tracks','ocean']
    set_4_labels = ['people','grass','clouds','bear','boats','jet','bridge','close-up','sun','ice']


    set_1_df = joined_df.loc[joined_df['image_labels'].apply(lambda labels: any(label in labels for label in set_1_labels))]
    set_1_df.loc[:, 'image_labels'] = set_1_df['image_labels'].apply(lambda labels: [label for label in labels if label in set_1_labels])
    set_1_df.reset_index(drop=True, inplace=True)

    set_2_df = joined_df.loc[joined_df['image_labels'].apply(lambda labels: any(label in labels for label in set_2_labels))]
    set_2_df.loc[:, 'image_labels'] = set_2_df['image_labels'].apply(lambda labels: [label for label in labels if label in set_2_labels])
    set_2_df.reset_index(drop=True, inplace=True)

    set_3_df = joined_df.loc[joined_df['image_labels'].apply(lambda labels: any(label in labels for label in set_3_labels))]
    set_3_df.loc[:, 'image_labels'] = set_3_df['image_labels'].apply(lambda labels: [label for label in labels if label in set_3_labels])
    set_3_df.reset_index(drop=True, inplace=True)

    set_4_df = joined_df.loc[joined_df['image_labels'].apply(lambda labels: any(label in labels for label in set_4_labels))]
    set_4_df.loc[:, 'image_labels'] = set_4_df['image_labels'].apply(lambda labels: [label for label in labels if label in set_4_labels])
    set_4_df.reset_index(drop=True, inplace=True)

    set_1_ds = Dataset.from_dict(set_1_df.to_dict(orient='list'))
    set_2_ds = Dataset.from_dict(set_2_df.to_dict(orient='list'))
    set_3_ds = Dataset.from_dict(set_3_df.to_dict(orient='list'))
    set_4_ds = Dataset.from_dict(set_4_df.to_dict(orient='list'))

    set_1_ds_partition = set_1_ds.train_test_split(test_size=0.2)
    set_2_ds_partition = set_2_ds.train_test_split(test_size=0.2)
    set_3_ds_partition = set_3_ds.train_test_split(test_size=0.2)
    set_4_ds_partition = set_4_ds.train_test_split(test_size=0.2)

    train_set_1_ds = set_1_ds_partition['train']
    test_set_1_ds = set_1_ds_partition['test']
    train_set_2_ds = set_2_ds_partition['train']
    test_set_2_ds = set_2_ds_partition['test']
    train_set_3_ds = set_3_ds_partition['train']
    test_set_3_ds = set_3_ds_partition['test']
    train_set_4_ds = set_4_ds_partition['train']
    test_set_4_ds = set_4_ds_partition['test']

    set_1_map = {'water': 0, 'buildings': 1, 'rocks': 2, 'field': 3, 'sand': 4, 'cars': 5, 'polar': 6, 'ruins': 7, 'statue': 8, 'wall': 9}
    set_2_map = {'sky': 0, 'mountain': 1, 'stone': 2, 'street': 3, 'birds': 4, 'leaf': 5, 'valley': 6, 'garden': 7, 'horses': 8, 'cat': 9}
    set_3_map = {'tree': 0, 'flowers': 1, 'snow': 2, 'plane': 3, 'beach': 4, 'plants': 5, 'house': 6, 'hills': 7, 'tracks': 8, 'ocean': 9}
    set_4_map = {'people': 0, 'grass': 1, 'clouds': 2, 'bear': 3, 'boats': 4, 'jet': 5, 'bridge': 6, 'close-up': 7, 'sun': 8, 'ice': 9}

    def remap_set_1_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(set_1_map[label])
        example["image_labels"] = mapped_labels
        return example

    def remap_set_2_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(set_2_map[label])
        example["image_labels"] = mapped_labels
        return example

    def remap_set_3_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(set_3_map[label])
        example["image_labels"] = mapped_labels
        return example

    def remap_set_4_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(set_4_map[label])
        example["image_labels"] = mapped_labels
        return example

    remapped_train_set_1_ds = train_set_1_ds.map(remap_set_1_labels)
    remapped_test_set_1_ds = test_set_1_ds.map(remap_set_1_labels)
    remapped_train_set_2_ds = train_set_2_ds.map(remap_set_2_labels)
    remapped_test_set_2_ds = test_set_2_ds.map(remap_set_2_labels)
    remapped_train_set_3_ds = train_set_3_ds.map(remap_set_3_labels)
    remapped_test_set_3_ds = test_set_3_ds.map(remap_set_3_labels)
    remapped_train_set_4_ds = train_set_4_ds.map(remap_set_4_labels)
    remapped_test_set_4_ds = test_set_4_ds.map(remap_set_4_labels)

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
    remapped_train_set_2_ds = remapped_train_set_2_ds.with_transform(apply_transforms)
    remapped_test_set_2_ds = remapped_test_set_2_ds.with_transform(apply_transforms)
    remapped_train_set_3_ds = remapped_train_set_3_ds.with_transform(apply_transforms)
    remapped_test_set_3_ds = remapped_test_set_3_ds.with_transform(apply_transforms)
    remapped_train_set_4_ds = remapped_train_set_4_ds.with_transform(apply_transforms)
    remapped_test_set_4_ds = remapped_test_set_4_ds.with_transform(apply_transforms)

    def collate_fn(batch):
        return {
            'image_name': torch.stack([x['image_name'] for x in batch]),
            'image_labels': torch.stack([x['image_labels'] for x in batch]).float()
        }

    set_1_train_loader = DataLoader(remapped_train_set_1_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)
    set_1_test_loader = DataLoader(remapped_test_set_1_ds, batch_size=64, drop_last=True, collate_fn=collate_fn)
    set_2_train_loader = DataLoader(remapped_train_set_2_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)
    set_2_test_loader = DataLoader(remapped_test_set_2_ds, batch_size=64, drop_last=True, collate_fn=collate_fn)
    set_3_train_loader = DataLoader(remapped_train_set_3_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)
    set_3_test_loader = DataLoader(remapped_test_set_3_ds, batch_size=64, drop_last=True, collate_fn=collate_fn)
    set_4_train_loader = DataLoader(remapped_train_set_4_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)
    set_4_test_loader = DataLoader(remapped_test_set_4_ds, batch_size=64, drop_last=True, collate_fn=collate_fn)

    print("TEST: ",next(iter(set_1_train_loader)))

main()