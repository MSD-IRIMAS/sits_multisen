"""Main file to run."""

import hydra
from omegaconf import DictConfig, OmegaConf

from main_analyses import main_analyses
from main_dataset import main_dataset


@hydra.main(config_name="hydra_config.yaml", config_path="config")
def main(args: DictConfig):
    """Run experiments.

    Main function to run experiments.

    Parameters
    ----------
    args: DictConfig
        The input configuration.

    Returns
    -------
    None
    """
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    if args.task == "dataset":
        main_dataset(args=args)
    elif args.task == "analyzes":
        main_analyses(args=args)
    else:
        raise ValueError("No task exist: " + args.task)


if __name__ == "__main__":
    main()