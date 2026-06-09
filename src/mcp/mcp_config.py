import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)


def _resolve_env(env: dict[str, str]) -> dict[str, str]:
    """Resuelve ${VAR} desde el entorno del proceso."""
    resolved = {}
    for key, value in env.items():
        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            resolved[key] = os.environ.get(var_name, "")
        else:
            resolved[key] = value
    return resolved


def load_mcp_config(path: str | Path = "mcp.json") -> list[MCPServerConfig]:
    config_path = Path(path)
    if not config_path.exists():
        return []

    with open(config_path) as f:
        raw = json.load(f)

    return [
        MCPServerConfig(
            name=server["name"],
            command=server["command"],
            args=server.get("args", []),
            env=_resolve_env(server.get("env", {})),
        )
        for server in raw.get("mcpServers", [])
    ]
