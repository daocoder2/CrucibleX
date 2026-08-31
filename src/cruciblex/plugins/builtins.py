from __future__ import annotations

import importlib

_REQUIRED_MODULES = (
    "cruciblex.runtime.generation.constraints",
    "cruciblex.plugins.generators.default",
    "cruciblex.plugins.generators.dump_replay",
    "cruciblex.plugins.executors.numpy",
    "cruciblex.plugins.comparators.allclose",
)
_OPTIONAL_MODULES = (
    "cruciblex.plugins.executors.aclnn",
    "cruciblex.plugins.executors.atb",
    "cruciblex.plugins.executors.temu",
    "cruciblex.plugins.executors.torch",
)


def load_builtin_plugins() -> None:
    for module_name in _REQUIRED_MODULES:
        importlib.import_module(module_name)
    for module_name in _OPTIONAL_MODULES:
        importlib.import_module(module_name)
