from __future__ import annotations

import importlib

_REQUIRED_MODULES = (
    "cruciblex.plugins.generators.default",
    "cruciblex.plugins.executors.numpy",
    "cruciblex.plugins.comparators.allclose",
)
_OPTIONAL_MODULES = (
    "cruciblex.plugins.executors.aclnn",
    "cruciblex.plugins.executors.torch",
)


def load_builtin_plugins() -> None:
    for module_name in _REQUIRED_MODULES:
        importlib.import_module(module_name)
    for module_name in _OPTIONAL_MODULES:
        importlib.import_module(module_name)
