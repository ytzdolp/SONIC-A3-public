#!/usr/bin/env python3
"""Train a small frozen-encoder G1-latent to A3-latent adapter.

Only this adapter is optimized.  The dataset already contains outputs from the
frozen G1 and A3 encoders, so neither SONIC nor the protected A3 Round2
checkpoint can be modified by this script.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vector_quantize_pytorch import FSQ


class LatentAdapter(nn.Module):
    """A deliberately small nonlinear map between two 64D latent spaces."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.module(latent.flatten(1)).reshape(-1, 2, 32)


def load_split(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        result = {
            key: torch.from_numpy(data[key].astype(np.float32)).to(device)
            for key in ("g1_latent", "a3_latent", "g1_token", "a3_token")
        }
        result["motion_id"] = torch.from_numpy(data["motion_id"].astype(np.int64)).to(device)
    if result["g1_latent"].shape[1:] != (2, 32):
        raise RuntimeError(f"{path}: unexpected latent shape {result['g1_latent'].shape}")
    return result


def cosine_stats(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    value = F.cosine_similarity(first.flatten(1), second.flatten(1), dim=1)
    return {
        "mean": float(value.mean()),
        "p05": float(torch.quantile(value, 0.05)),
        "minimum": float(value.min()),
    }


def metrics(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    predicted_token: torch.Tensor,
    target_token: torch.Tensor,
) -> dict:
    latent_delta = predicted_latent - target_latent
    token_delta = predicted_token - target_token
    return {
        "latent_rmse": float(torch.sqrt(torch.mean(latent_delta.square()))),
        "latent_mae": float(latent_delta.abs().mean()),
        "latent_cosine": cosine_stats(predicted_latent, target_latent),
        "token_rmse": float(torch.sqrt(torch.mean(token_delta.square()))),
        "token_mae": float(token_delta.abs().mean()),
        "token_cosine": cosine_stats(predicted_token, target_token),
        "token_element_exact_fraction": float(
            torch.mean((predicted_token == target_token).float())
        ),
        "token_window_exact_fraction": float(
            torch.mean(torch.all(predicted_token == target_token, dim=(1, 2)).float())
        ),
    }


def evaluate(
    model: LatentAdapter,
    split: dict[str, torch.Tensor],
    quantizer: FSQ,
) -> tuple[dict, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        predicted_latent = model(split["g1_latent"])
        predicted_token, _ = quantizer(predicted_latent)
    return (
        metrics(
            predicted_latent,
            split["a3_latent"],
            predicted_token,
            split["a3_token"],
        ),
        predicted_latent,
    )


def identity_metrics(split: dict[str, torch.Tensor]) -> dict:
    return metrics(
        split["g1_latent"],
        split["a3_latent"],
        split["g1_token"],
        split["a3_token"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--token-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    train = load_split(args.dataset_dir / "train.npz", device)
    validation = load_split(args.dataset_dir / "validation.npz", device)
    ood_path = args.dataset_dir / "ood.npz"
    ood = load_split(ood_path, device) if ood_path.exists() else None

    model = LatentAdapter(args.hidden_dim).to(device)
    quantizer = FSQ(levels=[32] * 32).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history = []
    best_validation_rmse = float("inf")
    best_epoch = 0
    best_state = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train["g1_latent"]), generator=generator)
        latent_loss_sum = 0.0
        token_loss_sum = 0.0
        samples = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            predicted_latent = model(train["g1_latent"][indices])
            predicted_token, _ = quantizer(predicted_latent)
            latent_loss = F.smooth_l1_loss(predicted_latent, train["a3_latent"][indices])
            token_loss = F.smooth_l1_loss(predicted_token, train["a3_token"][indices])
            loss = latent_loss + args.token_loss_weight * token_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(indices)
            latent_loss_sum += float(latent_loss.detach()) * count
            token_loss_sum += float(token_loss.detach()) * count
            samples += count

        validation_metrics, _ = evaluate(model, validation, quantizer)
        row = {
            "epoch": epoch,
            "train_latent_smooth_l1": latent_loss_sum / samples,
            "train_token_smooth_l1": token_loss_sum / samples,
            "validation_latent_rmse": validation_metrics["latent_rmse"],
            "validation_token_rmse": validation_metrics["token_rmse"],
            "validation_token_cosine": validation_metrics["token_cosine"]["mean"],
            "validation_token_element_exact_fraction": validation_metrics[
                "token_element_exact_fraction"
            ],
        }
        history.append(row)
        if validation_metrics["latent_rmse"] < best_validation_rmse:
            best_validation_rmse = validation_metrics["latent_rmse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} latent_loss={row['train_latent_smooth_l1']:.6f} "
                f"val_latent_rmse={row['validation_latent_rmse']:.6f} "
                f"val_token_cos={row['validation_token_cosine']:.6f} "
                f"val_token_exact={row['validation_token_element_exact_fraction']:.6f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    last_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    split_results = {}
    for name, split in (("train", train), ("validation", validation), ("ood", ood)):
        if split is None:
            continue
        adapted, _ = evaluate(model, split, quantizer)
        split_results[name] = {
            "identity_baseline": identity_metrics(split),
            "adapted": adapted,
        }

    checkpoint = {
        "schema_version": 1,
        "model_type": "G1ToA3LatentAdapter",
        "state_dict": best_state,
        "hidden_dim": args.hidden_dim,
        "best_epoch": best_epoch,
        "dataset_manifest": manifest,
        "training_config": vars(args),
        "results": split_results,
    }
    torch.save(checkpoint, args.output_dir / "best_validation.pt")
    torch.save(
        {**checkpoint, "state_dict": last_state, "checkpoint_kind": "last_epoch"},
        args.output_dir / "last.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = {
        "best_epoch": best_epoch,
        "completed_epochs": args.epochs,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "results": split_results,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
