"""Loads ``config/config.yaml``, expands ``${ENV_VAR:-default}`` references, applies
optional runtime overrides, and exposes it with dotted-key access (``cfg.get(
"chunking.chunk_size")``). No other module reads YAML or the environment directly."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from src.utils.exceptions import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Matches ${VAR} and ${VAR:-default}.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

_MISSING = object()


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: "re.Match[str]") -> str:
            var_name, default = match.group(1), match.group(2)
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class AppConfig:
    """Read-only view over the merged configuration dictionary."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "AppConfig":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse configuration: {exc}") from exc

        data = _expand_env(raw)
        if overrides:
            data = _deep_merge(data, overrides)
        return cls(data)

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Fetch a value by dotted key. Raises if absent and no default is given."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise ConfigError(f"Missing configuration key: '{dotted_key}'")
                return default
        return node

    def section(self, name: str) -> Dict[str, Any]:
        value = self._data.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration section '{name}' is not a mapping")
        return value

    def resolve_path(self, dotted_key: str) -> Path:
        """Resolve a path-valued config key relative to the project root."""
        raw = self.get(dotted_key)
        p = Path(raw)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "AppConfig":
        return AppConfig(_deep_merge(self._data, overrides))

    @property
    def raw(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"AppConfig(project={self.get('project.name', '?')!r})"
