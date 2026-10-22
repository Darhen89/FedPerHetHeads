"""Client implementation - can call FedPer and FedAvg clients."""

import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from fedper.constants import MEAN, STD
from fedper.implemented_models.mobile_model import MobileNetModelManager
from fedper.implemented_models.resnet_model import ResNetModelManager

PROJECT_DIR = Path(__file__).parent.parent.absolute()


class ClientDataloaders:
    """Client dataloaders."""

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
    ) -> None:
        """Initialize the client dataloaders."""
        self.trainloader = trainloader
        self.testloader = testloader


class ClientEssentials:
    """Client essentials."""

    def __init__(
        self,
        client_id: str,
        client_state_save_path: str = "",
    ) -> None:
        """Set client state save path and client ID."""

        self.client_id = int(client_id)
        self.client_state_save_path = (
            (client_state_save_path + f"/client_{self.client_id}")
            if client_state_save_path != ""
            else None
        )

class BaseClient(NumPyClient):
    """Implementation of Federated Averaging (FedAvg) Client."""

    def __init__(
        self,
        data_loaders: ClientDataloaders,
        config: DictConfig,
        client_essentials: ClientEssentials,
        model_manager_class: Union[
            Type[MobileNetModelManager], Type[ResNetModelManager]
        ],
    ):
        """Initialize client attributes.

        Args:
            config: dictionary containing the client configurations.
            client_id: id of the client.
            model_manager_class: class to be used as the model manager.
        """
        super().__init__()

        self.train_id = 1
        self.test_id = 1
        self.client_id = int(client_essentials.client_id)
        self.client_state_save_path = client_essentials.client_state_save_path
        self.hist: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.num_epochs: int = config["num_epochs"]
        self.model_manager = model_manager_class(
            client_id=self.client_id,
            config=config,
            trainloader=data_loaders.trainloader,
            testloader=data_loaders.testloader,
            client_save_path=self.client_state_save_path,
            learning_rate=config["learning_rate"],
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters."""
        return self.model_manager.model.get_parameters()

    def set_parameters(
        self, parameters: List[np.ndarray], evaluate: bool = False
    ) -> None:
        """Set the local model parameters to the received parameters.

        Args:
            parameters: parameters to set the model to.
        """
        _ = evaluate
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body") or k.startswith("_head")
        ]
        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(
        self,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local training to the whole model.

        Returns
        -------
            Dict with the train metrics.
        """
        epochs = self.num_epochs

        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()

        return self.model_manager.train(
            epochs=epochs,
        )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Train the provided parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
            Tuple containing the locally updated model parameters, \
                the number of examples used for training and \
                the training metrics.
        """
        self.set_parameters(parameters)

        train_results = self.perform_train()

        # Update train history
        self.hist[str(self.train_id)] = {
            **self.hist[str(self.train_id)],
            "trn": train_results,
        }
        print("<------- TRAIN RESULTS -------> :", train_results)

        self.train_id += 1

        return self.get_parameters(config), self.model_manager.train_dataset_size(), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Evaluate the provided global parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        self.set_parameters(parameters, evaluate=True)

        # Test the model
        tst_results = self.model_manager.test()
        print("<------- TEST RESULTS -------> :", tst_results)

        # Update test history
        self.hist[str(self.test_id)] = {
            **self.hist[str(self.test_id)],
            "tst": tst_results,
        }
        self.test_id += 1
        return (
            tst_results.get("loss", 0.0),
            self.model_manager.test_dataset_size(),
            {k: v for k, v in tst_results.items()},
        )
#{k: v for k, v in tst_results.items() if not isinstance(v, (dict, list))},

class FedPerClient(BaseClient):
    """Implementation of Federated Personalization (FedPer) Client."""
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local body parameters."""
        return [
            val.cpu().numpy()
            for _, val in self.model_manager.model.body.state_dict().items()
        ]

    def set_parameters(self, parameters: List[np.ndarray], evaluate=False) -> None:
        """Set the local body parameters to the received parameters.

        Args:
            parameters: parameters to set the body to.
            evaluate: whether the client is evaluating or not.
        """
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body")
        ]

        '''if not evaluate:
            # Only update client's local head if it hasn't trained yet
            print("Setting head parameters to global head parameters.")
            model_keys.extend(
                [
                    k
                    for k in self.model_manager.model.state_dict().keys()
                    if k.startswith("_head")
                ]
            )'''

        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)


def get_client_fn_simulation(
    config: DictConfig,
    client_state_save_path: str = "",
    set_1_train_loader: DataLoader = None,
    set_1_test_loader: DataLoader = None,
    set_2_train_loader: DataLoader = None,
    set_2_test_loader: DataLoader = None,
    set_3_train_loader: DataLoader = None,
    set_3_test_loader: DataLoader = None,
    set_4_train_loader: DataLoader = None,
    set_4_test_loader: DataLoader = None,
) -> Callable[[str], Client]:
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    model : DictConfig
        The model configuration.
    cleint_state_save_path : str
        The path to save the client state.

    Returns
    -------
    Tuple[Callable[[str], Client], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    set_1_train_loader=set_1_train_loader
    set_1_test_loader=set_1_test_loader
    set_2_train_loader=set_2_train_loader
    set_2_test_loader=set_2_test_loader
    set_3_train_loader=set_3_train_loader
    set_3_test_loader=set_3_test_loader
    set_4_train_loader=set_4_train_loader
    set_4_test_loader=set_4_test_loader

    def client_fn(cid: str) -> Client:
        """Create a Flower client representing a single organization."""
        cid_use = int(cid)
        if config.dataset.name.lower() == "corel5k":

            if cid_use == 0:
                train_loader = set_1_train_loader
                test_loader = set_1_test_loader
            elif cid_use == 1:
                train_loader = set_2_train_loader
                test_loader = set_2_test_loader
            elif cid_use == 2:
                train_loader = set_3_train_loader
                test_loader = set_3_test_loader
            elif cid_use == 3:
                train_loader = set_4_train_loader
                test_loader = set_4_test_loader

        manager: Union[Type[MobileNetModelManager], Type[ResNetModelManager]] = (
            MobileNetModelManager
        )
        if config.model_name.lower() == "resnet":
            manager = ResNetModelManager
        else:
            raise NotImplementedError("Model not implemented, check name.")
        client_data_loaders = ClientDataloaders(train_loader, test_loader)
        client_essentials = ClientEssentials(
            client_id=cid,
            client_state_save_path=client_state_save_path,
        )
        if client_state_save_path != "":
            return FedPerClient(
                data_loaders=client_data_loaders,
                client_essentials=client_essentials,
                config=config,
                model_manager_class=manager,
            ).to_client()
        return BaseClient(
            data_loaders=client_data_loaders,
            client_essentials=client_essentials,
            config=config,
            model_manager_class=manager,
        ).to_client()

    return client_fn
