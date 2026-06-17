import json
from pathlib import Path

from config.AppConfig import AppConfig


def load_config(path: str | Path = "config.json") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    with open(config_path) as f:
        raw = json.load(f)

    return AppConfig.model_validate(raw)
