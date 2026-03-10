from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict
import logging
import sys
import types

import torch


@dataclass
class ScGPTResources:
    repo_path: Path
    checkpoint_path: Path
    vocab_path: Path | None = None


def add_repo_to_path(repo_path: Path):
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    scgpt_pkg = repo_path / "scgpt"
    if scgpt_pkg.exists():
        existing = sys.modules.get("scgpt")
        if existing is None or not getattr(existing, "__path__", None):
            stub = types.ModuleType("scgpt")
            stub.__path__ = [str(scgpt_pkg)]
            stub.__package__ = "scgpt"
            logger = logging.getLogger("scGPT")
            if not logger.hasHandlers() or len(logger.handlers) == 0:
                logger.propagate = False
                logger.setLevel(logging.INFO)
                handler = logging.StreamHandler(sys.stdout)
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            stub.logger = logger
            sys.modules["scgpt"] = stub


def resolve_entrypoint(entrypoint: str):
    module_name, attr_name = entrypoint.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def strip_prefix(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not prefix:
        return state_dict
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped


def load_scgpt_model(
    entrypoint: str,
    repo_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
    model_args: Dict[str, Any] | None = None,
    prefix_to_strip: str | None = None,
):
    repo_path = Path(repo_path)
    checkpoint_path = Path(checkpoint_path)
    add_repo_to_path(repo_path)

    model_cls = resolve_entrypoint(entrypoint)
    model_args = model_args or {}
    model = model_cls(**model_args)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state
    else:
        state_dict = state

    if prefix_to_strip:
        state_dict = strip_prefix(state_dict, prefix_to_strip)

    try:
        from scgpt.utils import load_pretrained as scgpt_load_pretrained
    except Exception:
        scgpt_load_pretrained = None

    if scgpt_load_pretrained is not None:
        scgpt_load_pretrained(model, state_dict, verbose=False)
        missing, unexpected = [], []
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model, missing, unexpected
