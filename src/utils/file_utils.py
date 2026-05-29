"""
I/O utilities for reading and writing YAML and JSON configuration files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(yaml_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        yaml_path: Path to the YAML configuration file.
                   If relative, resolved against the current working directory.

    Returns:
        A dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    config_path = Path(yaml_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}\n"
            f"Make sure '{yaml_path}' exists in the project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path.resolve()}"
        )

    return config


def save_json(data: Dict[str, Any], json_path: str) -> None:
    """
    Save a dictionary as a JSON file (UTF-8, indented, non-ASCII preserved).

    Args:
        data: Dictionary to serialize.
        json_path: Destination file path (directories are created if needed).
    """
    output_path = Path(json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.

    Args:
        json_path: Path to the JSON file.

    Returns:
        A dictionary containing the parsed JSON data.

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
    """
    json_file = Path(json_path)

    if not json_file.exists():
        raise FileNotFoundError(
            f"JSON file not found: {json_file.resolve()}"
        )

    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)
