import yaml
import os
from pathlib import Path

def find_project_root(current_path: Path) -> Path:
    """
    Traverses upwards from the current path to find the project root directory.
    Assumes the project root is marked by the presence of a 'pyproject.toml' file.
    """
    # Iterate through all parent directories of the current path
    for parent in current_path.parents:
        # Check if 'pyproject.toml' file exists in the current parent directory
        if (parent / 'pyproject.toml').exists():
            return parent # If found, return this directory as the project root
    # If 'pyproject.toml' is not found after traversing all parent directories, raise a FileNotFoundError
    raise FileNotFoundError("Could not find project root directory (no pyproject.toml found in parent directories).")

def load_config(config_filename: str = 'config.yaml') -> dict:
    """
    Loads configuration from a YAML file located in the project root directory.

    Args:
        config_filename: The name of the YAML configuration file (e.g., 'config.yaml').

    Returns:
        A dictionary containing the loaded configuration.
    """
    # Get the directory of the current script (e.g., config_loader.py)
    current_script_dir = Path(__file__).parent

    # Find the project root directory based on a marker file (e.g., 'pyproject.toml')
    project_root = find_project_root(current_script_dir)
    
    # Construct the full path to the configuration file
    config_path = project_root / config_filename

    # Check if the configuration file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Open and load the YAML configuration file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config