from typing import Tuple
from omegaconf import OmegaConf


def load_config(file_path: str) -> Tuple[dict, str]:
    if not file_path.endswith(".yaml"):
        raise ValueError("The configuration file must be a yaml file")

    config = OmegaConf.load(file_path)

    base_config_path = config.get("base_config", "none")
    if base_config_path.lower() != "none":
        config_custom = config
        config = OmegaConf.load(base_config_path)
        config.merge_with(config_custom)

    config_str = OmegaConf.to_yaml(config)

    return config, config_str