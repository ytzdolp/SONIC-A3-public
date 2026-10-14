"""Runtime bridge from the released G1 encoder to the A3 token space."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from vector_quantize_pytorch import FSQ


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    import trl.trainer.utils as trainer_utils
    from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div

    trainer_utils.OnlineTrainerState = OnlineTrainerState
    trainer_utils.exact_div = exact_div
    return torch.load(path, map_location="cpu", weights_only=False)


def _make_mlp(state: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    indices = sorted(
        int(key.removeprefix(prefix).split(".", 1)[0])
        for key in state
        if key.startswith(prefix) and key.endswith(".weight")
    )
    if not indices:
        raise RuntimeError(f"checkpoint contains no tensors with prefix {prefix}")
    modules: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight = state[f"{prefix}{index}.weight"]
        bias = state[f"{prefix}{index}.bias"]
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)
        modules.append(layer)
        if position != len(indices) - 1:
            modules.append(nn.SiLU())
    return nn.Sequential(*modules)


class G1A3RuntimeAdapter(nn.Module):
    """Frozen G1 encoder followed by a trained 64D G1→A3 latent map and FSQ."""

    def __init__(
        self,
        g1_checkpoint: str | Path,
        adapter_checkpoint: str | Path,
        device: torch.device,
    ):
        super().__init__()
        g1_path = Path(g1_checkpoint)
        adapter_path = Path(adapter_checkpoint)
        g1_payload = _load_checkpoint(g1_path)
        adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)

        expected_g1_sha = (
            adapter_payload.get("dataset_manifest", {}).get("g1_checkpoint_sha256")
        )
        actual_g1_sha = _sha256(g1_path)
        if expected_g1_sha is not None and expected_g1_sha != actual_g1_sha:
            raise RuntimeError(
                "latent adapter was trained with a different G1 checkpoint: "
                f"expected={expected_g1_sha} actual={actual_g1_sha}"
            )

        self.g1_encoder = _make_mlp(
            g1_payload["policy_state_dict"], "actor_module.encoders.g1.module."
        )
        hidden_dim = int(adapter_payload["hidden_dim"])
        self.latent_adapter = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
        )
        adapter_state = {
            key.removeprefix("module."): value
            for key, value in adapter_payload["state_dict"].items()
        }
        self.latent_adapter.load_state_dict(adapter_state)
        self.quantizer = FSQ(levels=[32] * 32)
        self.g1_checkpoint_sha256 = actual_g1_sha
        self.adapter_checkpoint = str(adapter_path.resolve())
        self.to(device).eval().requires_grad_(False)

    def forward(self, obs_dict: dict[str, torch.Tensor], atm: nn.Module) -> torch.Tensor:
        """Return adapted FSQ tokens for the current tokenizer observation."""
        obs_with_sequence = {
            key: value.unsqueeze(1) if value.ndim == 2 else value
            for key, value in obs_dict.items()
        }
        tokenizer_obs = atm.parse_tokenizer_obs(obs_with_sequence)
        encoder_input = torch.cat(
            [
                tokenizer_obs["command_multi_future_nonflat"],
                tokenizer_obs["motion_anchor_ori_b_mf_nonflat"],
            ],
            dim=-1,
        ).reshape(next(iter(obs_dict.values())).shape[0], 640)
        g1_latent = self.g1_encoder(encoder_input).reshape(-1, 2, 32)
        adapted_latent = self.latent_adapter(g1_latent.flatten(1)).reshape(-1, 2, 32)
        tokens, _ = self.quantizer(adapted_latent)
        if not torch.isfinite(tokens).all():
            raise RuntimeError("G1→A3 runtime adapter produced non-finite tokens")
        return tokens
