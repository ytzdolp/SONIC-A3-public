# Upload manifest

## Included in the repository

- upstream SONIC training and motion-representation source required by A3 work;
- `gear_sonic` A3 modifications;
- `a3_locomotion` experimental task;
- A3 Hydra/network/reward/action configurations;
- `scripts/a3` conversion, validation, and training entry points;
- MotionBricks Python source and skeleton XML needed by upstream code;
- upstream license and legal notices, plus project-specific citation and
  security files;
- A3 introduction, progress, installation, asset-license, and release documents;
- third-party source/publication scope and HuMoR/PyTorch3D licenses;
- source/release validation tools.

## Added for the second snapshot

- public-safe rollout schemas, selective-data, and bounded action-adapter
  utilities that consume user-provided contract metadata;
- corrected-config native Dynamic Decoder utilities and generic training code;
- G1/A3 pair, frame-rate, joint-order, token, and checkpoint audit tools;
- ideal-reference and closed-loop latent-alignment dataset builders;
- frozen-decoder action evaluator and live G1-to-A3 runtime adapter;
- a sanitized results snapshot with no checkpoints, rollout data, or local paths;
- updated README, project introduction, progress history, and current plan.

## Deliberately excluded

- A3 URDF, MJCF, mesh, CAD, and vendor configuration;
- every external walking-policy repository, deployment program, original
  configuration, numerical contract, ONNX actor, CSV telemetry, and rollout;
- basketball repository and basketball motion data;
- all motion datasets under `sample_data` or local data directories;
- all SONIC, A3, and locomotion checkpoints;
- G1/A3 latent-adapter checkpoints and alignment NPZ files;
- `logs_rl`, `logs_eval`, TensorBoard data, videos, and screenshots;
- Python virtual environments and downloaded tools;
- local Docker/Isaac Lab copies;
- Holosoma source copy (install separately when required);
- unchanged G1 deployment, teleoperation SDK, and third-party binary trees;
- MotionBricks pretrained outputs, GIFs, and G1 mesh LFS objects;
- internal dialog handoff, PIDs, machine paths, hosts, and credentials.
- BONES-SEED archives, metadata copies, and converted G1 motions.

This manifest describes the clean public candidate. The complete experimental
workspace remains separate and private, with no shared Git history.

The current two-commit public history still contains the removed SMPLify-X
mapping and old machine-specific release-check literals. The working tree is
not a fully cleaned publication until that reachable history is rewritten (or
the relevant permission is obtained) and the remote is re-audited.

Project status is shared: README, introduction, progress, and results must use
the same accepted checkpoints, metrics, rejected branches, and capability
boundaries as the private research repository. Public wording may omit the
external numerical contract and private paths, but it must not preserve an old
or stronger progress claim.

## External dependencies

Reference these rather than copying them into this repository:

- SONIC upstream: `https://github.com/NVlabs/GR00T-WholeBodyControl`
- Isaac Lab: `https://github.com/isaac-sim/IsaacLab`
- AgiBot A3/A3U public robot models (Mulan PSL v2):
  `https://github.com/AgibotTech/A3-A3U-robot-model`
- Holosoma, when its retargeting pipeline is needed:
  `https://github.com/amazon-far/holosoma`
- BONES-SEED, after accepting its own license:
  `https://huggingface.co/datasets/bones-studio/seed`

Pin exact commits in experiment metadata when reproducibility matters.

## Before every push

```bash
python3 tools/verify_a3_source.py
python3 tools/check_release.py
```

Review `git status` and `git diff --cached --stat` manually before committing.
Also compare the current-result sections against the private repository before
each release; publication filtering is not permission for progress drift.
