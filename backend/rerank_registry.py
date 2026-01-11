from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BackendSpec:
    name: str
    module: str
    function: str
    conda_env: str | None = None


_BACKENDS: dict[str, BackendSpec] = {
    "langchain": BackendSpec(
        name="langchain",
        module="backend.langchain_rerank",
        function="langchain_llm_rerank",
        conda_env="lc",
    ),
}


def get_backend_spec(name: str) -> BackendSpec:
    key = (name or "").strip().lower()
    if key not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unsupported LLM rerank backend: {name}. Supported: {supported}")
    return _BACKENDS[key]


def load_backend(name: str) -> tuple[Callable[..., Any], BackendSpec]:
    spec = get_backend_spec(name)
    module = importlib.import_module(spec.module)
    handler = getattr(module, spec.function)
    return handler, spec


def list_backends() -> list[str]:
    return sorted(_BACKENDS)
