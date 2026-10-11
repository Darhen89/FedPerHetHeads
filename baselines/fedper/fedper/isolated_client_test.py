import hydraesp
from omegaconf import DictConfig
from datasets import load_dataset


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:

    # Load the dataset
    dataset = load_dataset("prithivMLmods/Multilabel-GeoSceneNet-16K")

    print("Sample: ",dataset["train"][0])

    # Extract unique labels
    labels = dataset["train"].features["label"].names

    # Create id2label mapping
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Print the mapping
    print(id2label)

main()