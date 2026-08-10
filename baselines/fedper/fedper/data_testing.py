import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

"""Load partition CIFAR10 data."""
# Only initialize `FederatedDataset` once

partitioner = IidPartitioner(num_partitions=1)
fds = FederatedDataset(
    dataset="uoft-cs/cifar10",
    partitioners={"train": partitioner},
)
partition = fds.load_partition(0)
# Divide data on each node: 80% train, 20% test
partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
# Construct dataloaders
partition_train_test = partition_train_test.with_transform(apply_transforms)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
labels_list_partition_train = [item['label'] for item in partition_train_test['train']]
labels_list_partition_test = [item['label'] for item in partition_train_test['test']]

animals = torch.tensor([2, 3, 4, 5, 6, 7])
animalsindicestrain = (torch.tensor(labels_list_partition_train)[..., None] == animals).any(-1).nonzero(as_tuple=True)[0]
animalsindicestest = (torch.tensor(labels_list_partition_test)[..., None] == animals).any(-1).nonzero(as_tuple=True)[0]
animalstrainsubset = torch.utils.data.Subset(partition_train_test['train'], animalsindicestrain)
animalstestsubset = torch.utils.data.Subset(partition_train_test['test'], animalsindicestest)

vehicles = torch.tensor([0, 1, 8, 9])
vehiclesindicestrain = (torch.tensor(labels_list_partition_train)[..., None] == vehicles).any(-1).nonzero(as_tuple=True)[0]
vehiclesindicestest = (torch.tensor(labels_list_partition_test)[..., None] == vehicles).any(-1).nonzero(as_tuple=True)[0]
vehiclestrainsubset = torch.utils.data.Subset(partition_train_test['train'], vehiclesindicestrain)
vehiclestestsubset = torch.utils.data.Subset(partition_train_test['test'], vehiclesindicestest)

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
animalsTrainDataset = RemappedLabels(animalstrainsubset, animal_map)
animalsTestDataset = RemappedLabels(animalstestsubset, animal_map)
vehiclesTrainDataset = RemappedLabels(vehiclestrainsubset, vehicle_map)
vehiclesTestDataset = RemappedLabels(vehiclestestsubset, vehicle_map)

animalstrainloader = DataLoader(animalsTrainDataset, batch_size=64, shuffle=True, num_workers=2)
animalstestloader = DataLoader(animalsTestDataset, batch_size=64)
vehiclestrainloader = DataLoader(vehiclesTrainDataset, batch_size=64, shuffle=True, num_workers=2)
vehiclestestloader = DataLoader(vehiclesTestDataset, batch_size=64)

print("Animals Train samples: ",next(iter(animalstrainloader))["label"])
print("Animals Test samples: ",next(iter(animalstestloader))["label"])
print("Vehicles Train samples: ",next(iter(vehiclestrainloader))["label"])
print("Vehicles Test samples: ",next(iter(vehiclestestloader))["label"])