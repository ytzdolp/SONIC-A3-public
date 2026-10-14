"""Single source of truth for the A3 residual-adapter data contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


SCHEMA_VERSION = 2
NUM_ACTIONS = 29
ADAPTER_INPUT_DIM = 1629
ADAPTER_PREFIX = "action_residual_adapter."
ADAPTER_JOINT_INDICES = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8)


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_actor_state(
    checkpoint: dict,
) -> tuple[str, dict[str, torch.Tensor]]:
    for key in ("actor_model_state_dict", "policy_state_dict"):
        if key in checkpoint:
            return key, checkpoint[key]
    raise RuntimeError(
        "checkpoint has neither actor_model_state_dict nor policy_state_dict"
    )


def base_policy_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash the deterministic frozen policy while excluding adapter/noise."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith(ADAPTER_PREFIX) or name in {"std", "log_std"}:
            continue
        value = state[name]
        if not torch.is_tensor(value):
            raise TypeError(f"policy state {name!r} is not a tensor")
        value = value.detach().cpu().contiguous()
        header = f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode()
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_base_policy_sha256(path: str | Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _, state = checkpoint_actor_state(checkpoint)
    return base_policy_sha256(state)
