# Install dependencies as needed:
# pip install kagglehub[hf-datasets]
import json
import os
from typing import Any

import torch
import kagglehub
from datasets import Dataset
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from matplotlib import pyplot as plt
from sympy import roots
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from webencodings import labels

import numpy.typing as npt

from PIL import Image
import numpy as np

#define dataset path
train_file_path = "./datasets/Corel-5k/train.json"
train_aug_file_path = "./datasets/Corel-5k/train_aug.json"
test_file_path = "./datasets/Corel-5k/test.json"
images_path = "datasets/Corel-5k/images"

#read the files to create pandas dataframe
with open(train_file_path) as json_data:
    train_data = json.load(json_data)
    train_df = pd.DataFrame(train_data['samples'])
    train_df['image_name'] = train_df['image_name'].apply(lambda image_name: Image.open(os.path.join(images_path, image_name)).convert('RGB'))

with open(train_aug_file_path) as json_data:
    train_aug_data = json.load(json_data)
    train_aug_df = pd.DataFrame(train_data['samples'])
    train_aug_df['image_name'] = train_aug_df['image_name'].apply(lambda image_name: Image.open(os.path.join(images_path, image_name)).convert('RGB'))

with open(test_file_path) as json_data:
    test_data = json.load(json_data)
    test_df = pd.DataFrame(test_data['samples'])
    test_df['image_name'] = test_df['image_name'].apply(lambda image_name: Image.open(os.path.join(images_path, image_name)).convert('RGB'))



joined_df = pd.concat([train_df,train_aug_df,test_df])

#print(joined_df["image_labels"].explode().value_counts())

water_df = joined_df.loc[joined_df['image_labels'].apply(lambda tup: 'water' in tup)]
water_df.reset_index(drop=True, inplace=True)
water_df.loc[:,'image_labels'] = water_df['image_labels'].apply(lambda elements: [label for label in elements if label != 'sky'])

sky_df = joined_df.loc[joined_df['image_labels'].apply(lambda tup: 'sky' in tup)]
sky_df.reset_index(drop=True, inplace=True)
sky_df.loc[:,'image_labels'] = sky_df['image_labels'].apply(lambda elements: [label for label in elements if label != 'water'])


water_ds = Dataset.from_dict(water_df.to_dict(orient='list'))
sky_ds = Dataset.from_dict(sky_df.to_dict(orient='list'))


water_partition = water_ds.train_test_split(test_size=0.2)
sky_partition = sky_ds.train_test_split(test_size=0.2)

pytorch_transforms = Compose([ToTensor(),Resize(size=(128,128)), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image_name"] = [pytorch_transforms(img) for img in batch["image_name"]]
    return batch

water_partition = water_partition.with_transform(apply_transforms)
sky_partition = sky_partition.with_transform(apply_transforms)

water_map = {'herd': 0,'tiger': 1,'reflection': 2,'cat': 3,'grass': 4,'harbor': 5,'sunrise': 6,'snow': 7,'face': 8,'museum': 9,'scotland': 10,'formation': 11,'bush': 12,'ocean': 13,'plane': 14,'branch': 15,'kauai': 16,'moose': 17,'calf': 18,'tree': 19,'frozen': 20,'grizzly': 21,'white-tailed': 22,'hut': 23,'fish': 24,'road': 25,'river': 26,'roofs': 27,'cave': 28,'valley': 29,'restaurant': 30,'elk': 31,'tusks': 32,'stone': 33,'temple': 34,'lighthouse': 35,'booby': 36,'market': 37,'desert': 38,'windmills': 39,'silhouette': 40,'interior': 41,'iguana': 42,'pool': 43,'sun': 44,'sunset': 45,'palace': 46,'hills': 47,'flight': 48,'decoration': 49,'swimmers': 50,'cottage': 51,'lizard': 52,'lake': 53,'locomotive': 54,'meadow': 55,'courtyard': 56,'mountain': 57,'close-up': 58,'shore': 59,'marine': 60,'door': 61,'cafe': 62,'people': 63,'cars': 64,'horizon': 65,'bear': 66,'garden': 67,'log': 68,'monastery': 69,'house': 70,'bengal': 71,'african': 72,'forest': 73,'sand': 74,'castle': 75,'palm': 76,'reefs': 77,'tulip': 78,'wood': 79,'canyon': 80,'city': 81,'shops': 82,'sign': 83,'train': 84,'black': 85,'pillar': 86,'dress': 87,'athlete': 88,'shrubs': 89,'waves': 90,'wall': 91,'peaks': 92,'hotel': 93,'vegetation': 94,'cubs': 95,'birds': 96,'rocks': 97,'clouds': 98,'elephant': 99,'shadows': 100,'porcupine': 101,'dock': 102,'lawn': 103,'light': 104,'monument': 105,'sea': 106,'gate': 107,'horns': 108,'fox': 109,'crab': 110,'zebra': 111,'polar': 112,'oahu': 113,'nets': 114,'lion': 115,'hillside': 116,'night': 117,'island': 118,'bulls': 119,'ruins': 120,'rodent': 121,'town': 122,'water': 123,'buildings': 124,'caribou': 125,'tundra': 126,'ships': 127,'art': 128,'antlers': 129,'clothes': 130,'coyote': 131,'statue': 132,'plants': 133,'window': 134,'canal': 135,'leaf': 136,'bridge': 137,'arctic': 138,'flowers': 139,'giraffe': 140,'nest': 141,'coast': 142,'tower': 143,'ground': 144,'sails': 145,'anemone': 146,'mist': 147,'maui': 148,'street': 149,'petals': 150,'display': 151,'park': 152,'skyline': 153,'field': 154,'frost': 155,'marsh': 156,'head': 157,'buddhist': 158,'church': 159,'fence': 160,'landscape': 161,'whales': 162,'dunes': 163,'arch': 164,'farms': 165,'flag': 166,'land': 167,'entrance': 168,'railroad': 169,'hawaii': 170,'ice': 171,'deer': 172,'sculpture': 173,'village': 174,'fountain': 175,'coral': 176,'boats': 177,'beach': 178}
sky_map = {'sand': 0,'temple': 1,'mountain': 2,'door': 3,'prop': 4,'stems': 5,'sunset': 6,'tracks': 7,'dock': 8,'waves': 9,'herd': 10,'zebra': 11,'tower': 12,'canal': 13,'entrance': 14,'forest': 15,'jet': 16,'beach': 17,'caribou': 18,'smoke': 19,'ground': 20,'stone': 21,'festival': 22,'flag': 23,'hotel': 24,'palace': 25,'buddha': 26,'hut': 27,'tiger': 28,'rocks': 29,'village': 30,'window': 31,'hills': 32,'elephant': 33,'black': 34,'river': 35,'land': 36,'gate': 37,'fence': 38,'shadows': 39,'lighthouse': 40,'kauai': 41,'street': 42,'field': 43,'goat': 44,'courtyard': 45,'flight': 46,'sign': 47,'buildings': 48,'bridge': 49,'albatross': 50,'wall': 51,'cottage': 52,'ruins': 53,'town': 54,'fountain': 55,'valley': 56,'leaf': 57,'island': 58,'light': 59,'roofs': 60,'vegetation': 61,'sun': 62,'clouds': 63,'tree': 64,'arch': 65,'deer': 66,'maui': 67,'buddhist': 68,'art': 69,'cat': 70,'pool': 71,'sculpture': 72,'coyote': 73,'cathedral': 74,'restaurant': 75,'statue': 76,'tulip': 77,'church': 78,'pillar': 79,'silhouette': 80,'frozen': 81,'baby': 82,'sky': 83,'antelope': 84,'monument': 85,'bush': 86,'ceremony': 87,'windmills': 88,'bulls': 89,'giraffe': 90,'lawn': 91,'post': 92,'ice': 93,'bear': 94,'head': 95,'park': 96,'landscape': 97,'museum': 98,'harbor': 99,'antlers': 100,'grass': 101,'dance': 102,'formation': 103,'snow': 104,'outside': 105,'trunk': 106,'plants': 107,'house': 108,'elk': 109,'pyramid': 110,'moose': 111,'stairs': 112,'lake': 113,'shrubs': 114,'barn': 115,'nest': 116,'desert': 117,'coast': 118,'market': 119,'sea': 120,'palm': 121,'hillside': 122,'balcony': 123,'close-up': 124,'flowers': 125,'frost': 126,'city': 127,'doorway': 128,'castle': 129,'locomotive': 130,'tundra': 131,'cars': 132,'decoration': 133,'face': 134,'ocean': 135,'monastery': 136,'boats': 137,'path': 138,'railroad': 139,'oahu': 140,'architecture': 141,'plane': 142,'branch': 143,'road': 144,'clothes': 145,'horizon': 146,'people': 147,'slope': 148,'canyon': 149,'lion': 150,'tusks': 151,'reflection': 152,'dunes': 153,'farms': 154,'garden': 155,'shore': 156,'ships': 157,'train': 158,'skyline': 159,'birds': 160,'wood': 161,'peaks': 162,'f-16': 163,'basket': 164,'fox': 165,'hawaii': 166,'vineyard': 167}

class RemappedLabels(Dataset):
    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        batch = self.dataset.__getitem__(idx)

        if len(batch) == 1:
            mapped_labels = []
            for label in batch["image_labels"]:
                mapped_labels.append(self.label_map[label])
            batch['image_labels'] = mapped_labels
            return batch
        else:
            mapped_instances = []
            for labels in batch["image_labels"]:
                mapped_labels = []
                for label in labels:
                    mapped_labels.append(self.label_map[label])
                mapped_instances.append(mapped_labels)
            batch['image_labels'] = mapped_instances
            return batch


water_train_ds = RemappedLabels(water_partition['train'], water_map)
sky_train_ds = RemappedLabels(sky_partition['train'], sky_map)
water_test_ds = RemappedLabels(water_partition['test'], water_map)
sky_test_ds = RemappedLabels(sky_partition['test'], sky_map)

water_train_loader = DataLoader(water_train_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=lambda x: x)
water_test_loader = DataLoader(sky_train_ds, batch_size=64, drop_last=True, collate_fn=lambda x: x)
sky_train_loader = DataLoader(water_test_ds, batch_size=64, shuffle=True, num_workers=2, drop_last=True, collate_fn=lambda x: x)
sky_test_loader = DataLoader(sky_test_ds, batch_size=64, drop_last=True, collate_fn=lambda x: x)


# Display image and label.
cosa = next(iter(water_train_loader))[0]

img = cosa['image_name']
label = cosa['image_labels']
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")




