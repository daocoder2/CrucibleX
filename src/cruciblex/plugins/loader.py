from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


class PluginLoadError(RuntimeError):
    pass


def load_plugins(paths: list[Path]) -> None:
    for path in paths:
        _load_path(path)


def _load_path(path: Path) -> None:
    if path.is_dir():
        for plugin_file in sorted(path.glob("*.py")):
            _load_file(plugin_file)
        return
    if path.is_file() and path.suffix == ".py":
        _load_file(path)
        return
    raise PluginLoadError(f"unsupported plugin path: {path}")


def _load_file(path: Path) -> None:
    module_name = f"cruciblex_user_plugin_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PluginLoadError(f"cannot load plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PluginLoadError(f"failed to load plugin {path}: {exc}") from exc
