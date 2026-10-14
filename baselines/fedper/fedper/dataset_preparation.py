"""Dataset preparation."""

import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import json
from PIL import Image

#from torch.utils.data import Dataset
from datasets import Dataset
from torchvision import transforms

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, \
    CenterCrop


class BaseDataset(Dataset):
    """Base class for all datasets."""

    def __init__(
        self,
        root: Path = Path(" /cifar10"),
        general_data_transform: transforms.transforms.Compose = None,
    ) -> None:
        """Initialize the dataset."""
        self.root = root
        self.classes = None
        self.data: torch.tensor = None
        self.targets: torch.tensor = None
        self.general_data_transform = general_data_transform

    def __getitem__(self, index):
        """Get the item at the given index."""
        data, targets = self.data[index], self.targets[index]
        if self.general_data_transform is not None:
            data = self.general_data_transform(data)
        return data, targets

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.targets)


class CIFAR10(BaseDataset):
    """CIFAR10 dataset."""

    def __init__(
        self,
        root: Path = Path("datasets/cifar10"),
        general_data_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.tensor(train_part.targets).long().squeeze()
        test_targets = torch.tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform

class Corel5k(BaseDataset):
    """CIFAR10 dataset."""

    def __init__(
        self,
        root: Path = Path("datasets/cifar10"),
        general_data_transform=None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.tensor(train_part.targets).long().squeeze()
        test_targets = torch.tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.general_data_transform = general_data_transform


def corel5k_preprocess(root, config):
    print("Preprocessing Corel-5k dataset...")
    # create a tmp folder to store the preprocessed data
    tmp_folder = Path(root, "tmp")
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)

    # remove any folder or file in tmp folder, even if it is not empty
    os.system(f"rm -rf {tmp_folder.as_posix()}/*")

    # get number of clients
    num_clients = config["num_clients"]

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

    print(joined_df["image_labels"].explode().value_counts())
    exit(0)

    water_df = joined_df.loc[joined_df['image_labels'].apply(lambda tup: 'water' in tup)]
    water_df.reset_index(drop=True, inplace=True)
    water_df.loc[:, 'image_labels'] = water_df['image_labels'].apply(
        lambda elements: [label for label in elements if label != 'sky'])

    sky_df = joined_df.loc[joined_df['image_labels'].apply(lambda tup: 'sky' in tup)]
    sky_df.reset_index(drop=True, inplace=True)
    sky_df.loc[:, 'image_labels'] = sky_df['image_labels'].apply(
        lambda elements: [label for label in elements if label != 'water'])

    water_ds = Dataset.from_dict(water_df.to_dict(orient='list'))
    sky_ds = Dataset.from_dict(sky_df.to_dict(orient='list'))

    water_partition = water_ds.train_test_split(test_size=0.2)
    sky_partition = sky_ds.train_test_split(test_size=0.2)

    water_map = {'herd': 0, 'tiger': 1, 'reflection': 2, 'cat': 3, 'grass': 4, 'harbor': 5, 'sunrise': 6, 'snow': 7,
                 'face': 8, 'museum': 9, 'scotland': 10, 'formation': 11, 'bush': 12, 'ocean': 13, 'plane': 14,
                 'branch': 15, 'kauai': 16, 'moose': 17, 'calf': 18, 'tree': 19, 'frozen': 20, 'grizzly': 21,
                 'white-tailed': 22, 'hut': 23, 'fish': 24, 'road': 25, 'river': 26, 'roofs': 27, 'cave': 28,
                 'valley': 29, 'restaurant': 30, 'elk': 31, 'tusks': 32, 'stone': 33, 'temple': 34, 'lighthouse': 35,
                 'booby': 36, 'market': 37, 'desert': 38, 'windmills': 39, 'silhouette': 40, 'interior': 41,
                 'iguana': 42, 'pool': 43, 'sun': 44, 'sunset': 45, 'palace': 46, 'hills': 47, 'flight': 48,
                 'decoration': 49, 'swimmers': 50, 'cottage': 51, 'lizard': 52, 'lake': 53, 'locomotive': 54,
                 'meadow': 55, 'courtyard': 56, 'mountain': 57, 'close-up': 58, 'shore': 59, 'marine': 60, 'door': 61,
                 'cafe': 62, 'people': 63, 'cars': 64, 'horizon': 65, 'bear': 66, 'garden': 67, 'log': 68,
                 'monastery': 69, 'house': 70, 'bengal': 71, 'african': 72, 'forest': 73, 'sand': 74, 'castle': 75,
                 'palm': 76, 'reefs': 77, 'tulip': 78, 'wood': 79, 'canyon': 80, 'city': 81, 'shops': 82, 'sign': 83,
                 'train': 84, 'black': 85, 'pillar': 86, 'dress': 87, 'athlete': 88, 'shrubs': 89, 'waves': 90,
                 'wall': 91, 'peaks': 92, 'hotel': 93, 'vegetation': 94, 'cubs': 95, 'birds': 96, 'rocks': 97,
                 'clouds': 98, 'elephant': 99, 'shadows': 100, 'porcupine': 101, 'dock': 102, 'lawn': 103, 'light': 104,
                 'monument': 105, 'sea': 106, 'gate': 107, 'horns': 108, 'fox': 109, 'crab': 110, 'zebra': 111,
                 'polar': 112, 'oahu': 113, 'nets': 114, 'lion': 115, 'hillside': 116, 'night': 117, 'island': 118,
                 'bulls': 119, 'ruins': 120, 'rodent': 121, 'town': 122, 'water': 123, 'buildings': 124, 'caribou': 125,
                 'tundra': 126, 'ships': 127, 'art': 128, 'antlers': 129, 'clothes': 130, 'coyote': 131, 'statue': 132,
                 'plants': 133, 'window': 134, 'canal': 135, 'leaf': 136, 'bridge': 137, 'arctic': 138, 'flowers': 139,
                 'giraffe': 140, 'nest': 141, 'coast': 142, 'tower': 143, 'ground': 144, 'sails': 145, 'anemone': 146,
                 'mist': 147, 'maui': 148, 'street': 149, 'petals': 150, 'display': 151, 'park': 152, 'skyline': 153,
                 'field': 154, 'frost': 155, 'marsh': 156, 'head': 157, 'buddhist': 158, 'church': 159, 'fence': 160,
                 'landscape': 161, 'whales': 162, 'dunes': 163, 'arch': 164, 'farms': 165, 'flag': 166, 'land': 167,
                 'entrance': 168, 'railroad': 169, 'hawaii': 170, 'ice': 171, 'deer': 172, 'sculpture': 173,
                 'village': 174, 'fountain': 175, 'coral': 176, 'boats': 177, 'beach': 178}
    sky_map = {'sand': 0, 'temple': 1, 'mountain': 2, 'door': 3, 'prop': 4, 'stems': 5, 'sunset': 6, 'tracks': 7,
               'dock': 8, 'waves': 9, 'herd': 10, 'zebra': 11, 'tower': 12, 'canal': 13, 'entrance': 14, 'forest': 15,
               'jet': 16, 'beach': 17, 'caribou': 18, 'smoke': 19, 'ground': 20, 'stone': 21, 'festival': 22,
               'flag': 23, 'hotel': 24, 'palace': 25, 'buddha': 26, 'hut': 27, 'tiger': 28, 'rocks': 29, 'village': 30,
               'window': 31, 'hills': 32, 'elephant': 33, 'black': 34, 'river': 35, 'land': 36, 'gate': 37, 'fence': 38,
               'shadows': 39, 'lighthouse': 40, 'kauai': 41, 'street': 42, 'field': 43, 'goat': 44, 'courtyard': 45,
               'flight': 46, 'sign': 47, 'buildings': 48, 'bridge': 49, 'albatross': 50, 'wall': 51, 'cottage': 52,
               'ruins': 53, 'town': 54, 'fountain': 55, 'valley': 56, 'leaf': 57, 'island': 58, 'light': 59,
               'roofs': 60, 'vegetation': 61, 'sun': 62, 'clouds': 63, 'tree': 64, 'arch': 65, 'deer': 66, 'maui': 67,
               'buddhist': 68, 'art': 69, 'cat': 70, 'pool': 71, 'sculpture': 72, 'coyote': 73, 'cathedral': 74,
               'restaurant': 75, 'statue': 76, 'tulip': 77, 'church': 78, 'pillar': 79, 'silhouette': 80, 'frozen': 81,
               'baby': 82, 'sky': 83, 'antelope': 84, 'monument': 85, 'bush': 86, 'ceremony': 87, 'windmills': 88,
               'bulls': 89, 'giraffe': 90, 'lawn': 91, 'post': 92, 'ice': 93, 'bear': 94, 'head': 95, 'park': 96,
               'landscape': 97, 'museum': 98, 'harbor': 99, 'antlers': 100, 'grass': 101, 'dance': 102,
               'formation': 103, 'snow': 104, 'outside': 105, 'trunk': 106, 'plants': 107, 'house': 108, 'elk': 109,
               'pyramid': 110, 'moose': 111, 'stairs': 112, 'lake': 113, 'shrubs': 114, 'barn': 115, 'nest': 116,
               'desert': 117, 'coast': 118, 'market': 119, 'sea': 120, 'palm': 121, 'hillside': 122, 'balcony': 123,
               'close-up': 124, 'flowers': 125, 'frost': 126, 'city': 127, 'doorway': 128, 'castle': 129,
               'locomotive': 130, 'tundra': 131, 'cars': 132, 'decoration': 133, 'face': 134, 'ocean': 135,
               'monastery': 136, 'boats': 137, 'path': 138, 'railroad': 139, 'oahu': 140, 'architecture': 141,
               'plane': 142, 'branch': 143, 'road': 144, 'clothes': 145, 'horizon': 146, 'people': 147, 'slope': 148,
               'canyon': 149, 'lion': 150, 'tusks': 151, 'reflection': 152, 'dunes': 153, 'farms': 154, 'garden': 155,
               'shore': 156, 'ships': 157, 'train': 158, 'skyline': 159, 'birds': 160, 'wood': 161, 'peaks': 162,
               'f-16': 163, 'basket': 164, 'fox': 165, 'hawaii': 166, 'vineyard': 167}

    train_water_ds = water_partition['train']
    test_water_ds = water_partition['test']
    train_sky_ds = sky_partition['train']
    test_sky_ds = sky_partition['test']

    def remap_water_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(water_map[label])
        # mapped_labels = mapped_labels + ([179] * (5 - len(mapped_labels)))
        example["image_labels"] = mapped_labels
        return example

    def remap_sky_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(sky_map[label])
        # mapped_labels = mapped_labels + ([168] * (5 - len(mapped_labels)))
        example["image_labels"] = mapped_labels
        return example

    remapped_train_water_ds = train_water_ds.map(remap_water_labels)
    remapped_test_water_ds = test_water_ds.map(remap_water_labels)
    remapped_train_sky_ds = train_sky_ds.map(remap_sky_labels)
    remapped_test_sky_ds = test_sky_ds.map(remap_sky_labels)

    pytorch_transforms = Compose([ToTensor(),
                                  Resize(size=(32, 32)),
                                  RandomHorizontalFlip(),
                                  RandomRotation(30),
                                  CenterCrop(size=(32, 32)),
                                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_water_transforms(batch):
        if isinstance(batch['image_name'], list):
            batch['image_name'] = torch.stack([pytorch_transforms(img) for img in batch['image_name']])
        else:
            batch['image_name'] = pytorch_transforms(batch['image'])
        # One-hot encode the multilabels
        all_labels = [torch.tensor(labels) for labels in batch['image_labels']]
        # Create one-hot encoding for each image's classes
        one_hot_labels = [nn.functional.one_hot(label, num_classes=179).sum(dim=0) for label in all_labels]
        # Stack them into a batch
        batch['image_labels'] = torch.stack(one_hot_labels)

        return batch

    def apply_sky_transforms(batch):
        if isinstance(batch['image_name'], list):
            batch['image_name'] = torch.stack([pytorch_transforms(img) for img in batch['image_name']])
        else:
            batch['image_name'] = pytorch_transforms(batch['image'])
        # One-hot encode the multilabels
        all_labels = [torch.tensor(labels) for labels in batch['image_labels']]
        # Create one-hot encoding for each image's classes
        one_hot_labels = [nn.functional.one_hot(label, num_classes=168).sum(dim=0) for label in all_labels]
        # Stack them into a batch
        batch['image_labels'] = torch.stack(one_hot_labels)

        return batch

    remapped_train_water_ds = remapped_train_water_ds.with_transform(apply_water_transforms)
    remapped_test_water_ds = remapped_test_water_ds.with_transform(apply_water_transforms)
    remapped_train_sky_ds = remapped_train_sky_ds.with_transform(apply_sky_transforms)
    remapped_test_sky_ds = remapped_test_sky_ds.with_transform(apply_sky_transforms)

    def collate_fn(batch):
        return {
            'image_name': torch.stack([x['image_name'] for x in batch]),
            'image_labels': torch.stack([x['image_labels'] for x in batch]).float()
        }

    # {'image_name': tensor([[IMG1],[IMG2]]), 'label': tensor([[ETI1],[ETI2]])}

    water_train_loader = DataLoader(remapped_train_water_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)
    water_test_loader = DataLoader(remapped_test_water_ds, batch_size=64, drop_last=True, collate_fn=collate_fn)
    sky_train_loader = DataLoader(remapped_train_sky_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=collate_fn)
    sky_test_loader = DataLoader(remapped_test_sky_ds, batch_size=64, drop_last=True, collate_fn=collate_fn)

    return water_train_loader, water_test_loader, sky_train_loader, sky_test_loader



def call_dataset(dataset_name, root, **kwargs):
    """Call the dataset."""
    if dataset_name == "cifar10":
        return CIFAR10(root, **kwargs)
    elif dataset_name == "corel5k":
        return Corel5k(root, **kwargs)
    raise ValueError(f"Dataset {dataset_name} not supported.")

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

fds = None  # Cache FederatedDataset

def dataset_class_type_split(config, cid):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=config["num_clients"])
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(cid)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    labels_list_partition_train = [item['label'] for item in partition_train_test['train']]
    labels_list_partition_test = [item['label'] for item in partition_train_test['test']]

    animals = torch.tensor([2, 3, 4, 5, 6, 7])
    animals_indices_train = (torch.tensor(labels_list_partition_train)[..., None] == animals).any(-1).nonzero(as_tuple=True)[0]
    animals_indices_test = (torch.tensor(labels_list_partition_test)[..., None] == animals).any(-1).nonzero(as_tuple=True)[0]
    animals_train_subset = torch.utils.data.Subset(partition_train_test['train'], animals_indices_train)
    animals_test_subset = torch.utils.data.Subset(partition_train_test['test'], animals_indices_test)

    vehicles = torch.tensor([0, 1, 8, 9])
    vehicles_indices_train = (torch.tensor(labels_list_partition_train)[..., None] == vehicles).any(-1).nonzero(as_tuple=True)[0]
    vehicles_indices_test = (torch.tensor(labels_list_partition_test)[..., None] == vehicles).any(-1).nonzero(as_tuple=True)[0]
    vehicles_train_subset = torch.utils.data.Subset(partition_train_test['train'], vehicles_indices_train)
    vehicles_test_subset = torch.utils.data.Subset(partition_train_test['test'], vehicles_indices_test)

    class RemappedLabels(torch.utils.data.Dataset):
        def __init__(self, dataset, label_map):
            self.dataset = dataset
            self.label_map = label_map

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            # The DataLoader passes an integer idx to RemappedLabels.
            # If self.dataset is a torch.utils.data.Subset and its indices are tensors,
            # accessing self.dataset[idx] will cause a TypeError in the underlying dataset
            # because Subset will pass a 0-d tensor as an index.
            if isinstance(self.dataset, torch.utils.data.Subset) and isinstance(self.dataset.indices, torch.Tensor):
                # Extract the actual index from the Subset's tensor indices and convert to a Python int
                actual_idx = int(self.dataset.indices[idx])
                # Access the underlying dataset directly with the integer index
                sample = dict(self.dataset.dataset[actual_idx])
            else:
                # For other dataset types, or Subset with list/tuple indices, proceed normally
                sample = dict(self.dataset[idx])

            sample["label"] = self.label_map[int(sample["label"])]
            return sample

    animal_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
    vehicle_map = {0: 0, 1: 1, 8: 2, 9: 3}

    animals_train_dataset = RemappedLabels(animals_train_subset, animal_map)
    animals_test_dataset = RemappedLabels(animals_test_subset, animal_map)
    vehicles_train_dataset = RemappedLabels(vehicles_train_subset, vehicle_map)
    vehicles_test_dataset = RemappedLabels(vehicles_test_subset, vehicle_map)

    animals_train_loader = DataLoader(animals_train_dataset, batch_size=64, shuffle=True, num_workers=2)
    animals_test_loader = DataLoader(animals_test_dataset, batch_size=64)
    vehicles_train_loader = DataLoader(vehicles_train_dataset, batch_size=64, shuffle=True, num_workers=2)
    vehicles_test_loader = DataLoader(vehicles_test_dataset, batch_size=64)

    return animals_train_loader, animals_test_loader, vehicles_train_loader, vehicles_test_loader

def corel5k_dataset_split(config, cid):
    # define dataset path
    train_file_path = "./datasets/Corel-5k/train.json"
    train_aug_file_path = "./datasets/Corel-5k/train_aug.json"
    test_file_path = "./datasets/Corel-5k/test.json"
    images_path = "datasets/Corel-5k/images"

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

    # print(joined_df["image_labels"].explode().value_counts())

    water_df = joined_df.loc[joined_df['image_labels'].apply(lambda tup: 'water' in tup)]
    water_df.reset_index(drop=True, inplace=True)
    water_df.loc[:, 'image_labels'] = water_df['image_labels'].apply(
        lambda elements: [label for label in elements if label != 'sky'])

    sky_df = joined_df.loc[joined_df['image_labels'].apply(lambda tup: 'sky' in tup)]
    sky_df.reset_index(drop=True, inplace=True)
    sky_df.loc[:, 'image_labels'] = sky_df['image_labels'].apply(
        lambda elements: [label for label in elements if label != 'water'])

    water_ds = Dataset.from_dict(water_df.to_dict(orient='list'))
    sky_ds = Dataset.from_dict(sky_df.to_dict(orient='list'))

    water_partition = water_ds.train_test_split(test_size=0.2)
    sky_partition = sky_ds.train_test_split(test_size=0.2)

    water_map = {'herd': 0, 'tiger': 1, 'reflection': 2, 'cat': 3, 'grass': 4, 'harbor': 5, 'sunrise': 6, 'snow': 7,
                 'face': 8, 'museum': 9, 'scotland': 10, 'formation': 11, 'bush': 12, 'ocean': 13, 'plane': 14,
                 'branch': 15, 'kauai': 16, 'moose': 17, 'calf': 18, 'tree': 19, 'frozen': 20, 'grizzly': 21,
                 'white-tailed': 22, 'hut': 23, 'fish': 24, 'road': 25, 'river': 26, 'roofs': 27, 'cave': 28,
                 'valley': 29, 'restaurant': 30, 'elk': 31, 'tusks': 32, 'stone': 33, 'temple': 34, 'lighthouse': 35,
                 'booby': 36, 'market': 37, 'desert': 38, 'windmills': 39, 'silhouette': 40, 'interior': 41,
                 'iguana': 42, 'pool': 43, 'sun': 44, 'sunset': 45, 'palace': 46, 'hills': 47, 'flight': 48,
                 'decoration': 49, 'swimmers': 50, 'cottage': 51, 'lizard': 52, 'lake': 53, 'locomotive': 54,
                 'meadow': 55, 'courtyard': 56, 'mountain': 57, 'close-up': 58, 'shore': 59, 'marine': 60, 'door': 61,
                 'cafe': 62, 'people': 63, 'cars': 64, 'horizon': 65, 'bear': 66, 'garden': 67, 'log': 68,
                 'monastery': 69, 'house': 70, 'bengal': 71, 'african': 72, 'forest': 73, 'sand': 74, 'castle': 75,
                 'palm': 76, 'reefs': 77, 'tulip': 78, 'wood': 79, 'canyon': 80, 'city': 81, 'shops': 82, 'sign': 83,
                 'train': 84, 'black': 85, 'pillar': 86, 'dress': 87, 'athlete': 88, 'shrubs': 89, 'waves': 90,
                 'wall': 91, 'peaks': 92, 'hotel': 93, 'vegetation': 94, 'cubs': 95, 'birds': 96, 'rocks': 97,
                 'clouds': 98, 'elephant': 99, 'shadows': 100, 'porcupine': 101, 'dock': 102, 'lawn': 103, 'light': 104,
                 'monument': 105, 'sea': 106, 'gate': 107, 'horns': 108, 'fox': 109, 'crab': 110, 'zebra': 111,
                 'polar': 112, 'oahu': 113, 'nets': 114, 'lion': 115, 'hillside': 116, 'night': 117, 'island': 118,
                 'bulls': 119, 'ruins': 120, 'rodent': 121, 'town': 122, 'water': 123, 'buildings': 124, 'caribou': 125,
                 'tundra': 126, 'ships': 127, 'art': 128, 'antlers': 129, 'clothes': 130, 'coyote': 131, 'statue': 132,
                 'plants': 133, 'window': 134, 'canal': 135, 'leaf': 136, 'bridge': 137, 'arctic': 138, 'flowers': 139,
                 'giraffe': 140, 'nest': 141, 'coast': 142, 'tower': 143, 'ground': 144, 'sails': 145, 'anemone': 146,
                 'mist': 147, 'maui': 148, 'street': 149, 'petals': 150, 'display': 151, 'park': 152, 'skyline': 153,
                 'field': 154, 'frost': 155, 'marsh': 156, 'head': 157, 'buddhist': 158, 'church': 159, 'fence': 160,
                 'landscape': 161, 'whales': 162, 'dunes': 163, 'arch': 164, 'farms': 165, 'flag': 166, 'land': 167,
                 'entrance': 168, 'railroad': 169, 'hawaii': 170, 'ice': 171, 'deer': 172, 'sculpture': 173,
                 'village': 174, 'fountain': 175, 'coral': 176, 'boats': 177, 'beach': 178}
    sky_map = {'sand': 0, 'temple': 1, 'mountain': 2, 'door': 3, 'prop': 4, 'stems': 5, 'sunset': 6, 'tracks': 7,
               'dock': 8, 'waves': 9, 'herd': 10, 'zebra': 11, 'tower': 12, 'canal': 13, 'entrance': 14, 'forest': 15,
               'jet': 16, 'beach': 17, 'caribou': 18, 'smoke': 19, 'ground': 20, 'stone': 21, 'festival': 22,
               'flag': 23, 'hotel': 24, 'palace': 25, 'buddha': 26, 'hut': 27, 'tiger': 28, 'rocks': 29, 'village': 30,
               'window': 31, 'hills': 32, 'elephant': 33, 'black': 34, 'river': 35, 'land': 36, 'gate': 37, 'fence': 38,
               'shadows': 39, 'lighthouse': 40, 'kauai': 41, 'street': 42, 'field': 43, 'goat': 44, 'courtyard': 45,
               'flight': 46, 'sign': 47, 'buildings': 48, 'bridge': 49, 'albatross': 50, 'wall': 51, 'cottage': 52,
               'ruins': 53, 'town': 54, 'fountain': 55, 'valley': 56, 'leaf': 57, 'island': 58, 'light': 59,
               'roofs': 60, 'vegetation': 61, 'sun': 62, 'clouds': 63, 'tree': 64, 'arch': 65, 'deer': 66, 'maui': 67,
               'buddhist': 68, 'art': 69, 'cat': 70, 'pool': 71, 'sculpture': 72, 'coyote': 73, 'cathedral': 74,
               'restaurant': 75, 'statue': 76, 'tulip': 77, 'church': 78, 'pillar': 79, 'silhouette': 80, 'frozen': 81,
               'baby': 82, 'sky': 83, 'antelope': 84, 'monument': 85, 'bush': 86, 'ceremony': 87, 'windmills': 88,
               'bulls': 89, 'giraffe': 90, 'lawn': 91, 'post': 92, 'ice': 93, 'bear': 94, 'head': 95, 'park': 96,
               'landscape': 97, 'museum': 98, 'harbor': 99, 'antlers': 100, 'grass': 101, 'dance': 102,
               'formation': 103, 'snow': 104, 'outside': 105, 'trunk': 106, 'plants': 107, 'house': 108, 'elk': 109,
               'pyramid': 110, 'moose': 111, 'stairs': 112, 'lake': 113, 'shrubs': 114, 'barn': 115, 'nest': 116,
               'desert': 117, 'coast': 118, 'market': 119, 'sea': 120, 'palm': 121, 'hillside': 122, 'balcony': 123,
               'close-up': 124, 'flowers': 125, 'frost': 126, 'city': 127, 'doorway': 128, 'castle': 129,
               'locomotive': 130, 'tundra': 131, 'cars': 132, 'decoration': 133, 'face': 134, 'ocean': 135,
               'monastery': 136, 'boats': 137, 'path': 138, 'railroad': 139, 'oahu': 140, 'architecture': 141,
               'plane': 142, 'branch': 143, 'road': 144, 'clothes': 145, 'horizon': 146, 'people': 147, 'slope': 148,
               'canyon': 149, 'lion': 150, 'tusks': 151, 'reflection': 152, 'dunes': 153, 'farms': 154, 'garden': 155,
               'shore': 156, 'ships': 157, 'train': 158, 'skyline': 159, 'birds': 160, 'wood': 161, 'peaks': 162,
               'f-16': 163, 'basket': 164, 'fox': 165, 'hawaii': 166, 'vineyard': 167}

    train_water_ds = water_partition['train']
    test_water_ds = water_partition['test']
    train_sky_ds = sky_partition['train']
    test_sky_ds = sky_partition['test']

    def remap_water_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(water_map[label])
        # mapped_labels = mapped_labels + ([179] * (5 - len(mapped_labels)))
        example["image_labels"] = mapped_labels
        return example

    def remap_sky_labels(example):
        mapped_labels = []
        for label in example["image_labels"]:
            mapped_labels.append(sky_map[label])
        # mapped_labels = mapped_labels + ([168] * (5 - len(mapped_labels)))
        example["image_labels"] = mapped_labels
        return example

    remapped_train_water_ds = train_water_ds.map(remap_water_labels)
    remapped_test_water_ds = test_water_ds.map(remap_water_labels)
    remapped_train_sky_ds = train_sky_ds.map(remap_sky_labels)
    remapped_test_sky_ds = test_sky_ds.map(remap_sky_labels)

    pytorch_transforms = Compose([ToTensor(),
                                  Resize(size=(32, 32)),
                                  RandomHorizontalFlip(),
                                  RandomRotation(30),
                                  CenterCrop(size=(32, 32)),
                                  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_water_transforms(batch):
        if isinstance(batch['image_name'], list):
            batch['image_name'] = torch.stack([pytorch_transforms(img) for img in batch['image_name']])
        else:
            batch['pixel_values'] = pytorch_transforms(batch['image'])
        # One-hot encode the multilabels
        all_labels = [torch.tensor(labels) for labels in batch['image_labels']]
        # Create one-hot encoding for each image's classes
        one_hot_labels = [nn.functional.one_hot(label, num_classes=179).sum(dim=0) for label in all_labels]
        # Stack them into a batch
        batch['image_labels'] = torch.stack(one_hot_labels)

        return batch

    def apply_sky_transforms(batch):
        if isinstance(batch['image_name'], list):
            batch['image_name'] = torch.stack([pytorch_transforms(img) for img in batch['image_name']])
        else:
            batch['pixel_values'] = pytorch_transforms(batch['image'])
        # One-hot encode the multilabels
        all_labels = [torch.tensor(labels) for labels in batch['image_labels']]
        # Create one-hot encoding for each image's classes
        one_hot_labels = [nn.functional.one_hot(label, num_classes=168).sum(dim=0) for label in all_labels]
        # Stack them into a batch
        batch['image_labels'] = torch.stack(one_hot_labels)

        return batch

    remapped_train_water_ds = remapped_train_water_ds.with_transform(apply_water_transforms)
    remapped_test_water_ds = remapped_test_water_ds.with_transform(apply_water_transforms)
    remapped_train_sky_ds = remapped_train_sky_ds.with_transform(apply_sky_transforms)
    remapped_test_sky_ds = remapped_test_sky_ds.with_transform(apply_sky_transforms)

    def collate_fn(batch):
        return {
            'image_name': torch.stack([x['image_name'] for x in batch]),
            'image_labels': torch.stack([x['image_labels'] for x in batch]).float()
        }

    # {'image_name': tensor([[IMG1],[IMG2]]), 'label': tensor([[ETI1],[ETI2]])}

    water_train_loader = DataLoader(remapped_train_water_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True,
                                    collate_fn=collate_fn)
    water_test_loader = DataLoader(remapped_test_water_ds, batch_size=64, drop_last=True)
    sky_train_loader = DataLoader(remapped_train_sky_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    sky_test_loader = DataLoader(remapped_test_sky_ds, batch_size=64, drop_last=True)

    return water_train_loader, water_test_loader, sky_train_loader, sky_test_loader

def randomly_assign_classes(
    dataset: Dataset, client_num: int, class_num: int
) -> Dict[str, Union[Dict[Any, Any], List[Any]]]:
    # ) -> Dict[str, Any]:
    """Randomly assign number classes to clients."""

    print("Before Dataset classes: ",dataset.classes)

    partition: Dict[str, Union[Dict, List]] = {"separation": {}, "data_indices": []}
    data_indices: List[List[int]] = [[] for _ in range(client_num)]
    targets_numpy = np.array(dataset.targets, dtype=np.int32)
    label_list = list(range(len(dataset.classes)))

    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0].tolist() for i in label_list
    ]

    assigned_labels = []
    selected_times = [0 for _ in label_list]
    for _ in range(client_num):
        sampled_labels = random.sample(label_list, class_num)
        assigned_labels.append(sampled_labels)
        for j in sampled_labels:
            selected_times[j] += 1

    batch_sizes = _get_batch_sizes(
        targets_numpy=targets_numpy,
        label_list=label_list,
        selected_times=selected_times,
    )

    data_indices = _get_data_indices(
        batch_sizes=batch_sizes,
        data_indices=data_indices,
        data_idx_for_each_label=data_idx_for_each_label,
        assigned_labels=assigned_labels,
        client_num=client_num,
    )

    partition["data_indices"] = data_indices

    return partition  # , stats


def _get_batch_sizes(
    targets_numpy: np.ndarray,
    label_list: List[int],
    selected_times: List[int],
) -> np.ndarray:
    """Get batch sizes for each label."""
    labels_count = Counter(targets_numpy)
    batch_sizes = np.zeros_like(label_list)
    for i in label_list:
        print(f"label: {i}, count: {labels_count[i]}")
        print(f"selected times: {selected_times[i]}")
        batch_sizes[i] = int(labels_count[i] / selected_times[i])

    return batch_sizes


def _get_data_indices(
    batch_sizes: np.ndarray,
    data_indices: List[List[int]],
    data_idx_for_each_label: List[List[int]],
    assigned_labels: List[List[int]],
    client_num: int,
) -> List[List[int]]:
    for i in range(client_num):
        for cls in assigned_labels[i]:
            if len(data_idx_for_each_label[cls]) < 2 * batch_sizes[cls]:
                batch_size = len(data_idx_for_each_label[cls])
            else:
                batch_size = batch_sizes[cls]
            selected_idx = random.sample(data_idx_for_each_label[cls], batch_size)
            data_indices_use: np.ndarray = np.concatenate(
                [data_indices[i], selected_idx], axis=0
            ).astype(np.int64)
            data_indices[i] = data_indices_use.tolist()
            # data_indices[i]: np.ndarray = np.concatenate(
            #    [data_indices[i], selected_idx], axis=0
            # ).astype(np.int64)
            data_idx_for_each_label[cls] = list(
                set(data_idx_for_each_label[cls]) - set(selected_idx)
            )

        data_indices[i] = data_indices[i]

    return data_indices
