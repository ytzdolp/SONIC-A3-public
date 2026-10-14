# SONIC-A3 installation

## Supported research setup

The current A3 training code was developed around:

- Ubuntu 22.04 or newer;
- NVIDIA GPU and compatible driver;
- Isaac Sim with Isaac Lab 2.3+;
- Python 3.11 inside the Isaac Lab environment;
- MuJoCo only for asset/order and sim-to-sim work.

Isaac Lab and MuJoCo serve different purposes. MuJoCo is optional for the first
Isaac Lab source check and is not a replacement for Isaac Lab PPO training.

## 1. Clone

For an independent repository:

```bash
git clone <your-repository-url> SONIC-A3
cd SONIC-A3
git remote add upstream https://github.com/NVlabs/GR00T-WholeBodyControl.git
```

This repository does not require Git LFS unless you deliberately add separately
licensed assets or checkpoints. Avoid adding them to ordinary Git history.

## 2. Install Isaac Sim and Isaac Lab

Follow the official Isaac Lab installation guide and select a release compatible
with Isaac Lab 2.3+. Verify from the same Python environment that will run SONIC:

```bash
python -c "import isaaclab; print(isaaclab.__version__)"
```

If Isaac Lab provides a launcher such as `isaaclab.sh -p`, use that launcher in
place of `python` for all commands below.

## 3. Install SONIC training dependencies

From the repository root, using the Isaac Lab Python environment:

```bash
python -m pip install -e "gear_sonic[training]"
```

Optional MuJoCo tools should use a separate lightweight environment when they do
not need Isaac Lab:

```bash
python3 -m venv .venvs/mujoco-tools
source .venvs/mujoco-tools/bin/activate
python -m pip install --upgrade pip
python -m pip install mujoco numpy scipy pyyaml
```

Do not copy a virtual environment into the repository or share one environment
between incompatible Isaac Sim and standalone Python installations.

## 4. Provide A3 assets

The repository does not duplicate A3 robot assets. Obtain the public A3/A3U
models from `https://github.com/AgibotTech/A3-A3U-robot-model`, comply with its
Mulan PSL v2 license, select the revision matching your robot, and create this
project layout:

```text
gear_sonic/data/assets/robot_description/
├── urdf/a3/urdf/a3.urdf
├── urdf/a3/meshes/...
└── mjcf/a3.xml
```

The URDF is used by Isaac Lab. The MJCF is used for motion-library and MuJoCo
order work. Paths are case-sensitive.

Before using an asset, confirm:

- all 29 controlled joint names match `gear_sonic/envs/manager_env/robots/a3.py`;
- mesh paths resolve;
- joint axes, limits, default pose, and units are correct;
- user-facing Isaac Lab stiffness/damping remain in SI units and are not
  multiplied by `180/pi`;
- you retained the robot-model license and required notices.

## 5. Provide motion data only when needed

Training configs refer to local files below `sample_data/`. Those files are not
distributed. Generate or place data locally only after verifying its permitted
uses, for example:

```text
sample_data/
├── a3_stage1_training/a3_stage1_kinematic_train.pkl
└── a3_stage1_training/a3_teacher_candidates_validated.pkl
```

Keep `sample_data/` ignored. Prefer a manifest containing filenames, hashes,
frame rates, joint order, provenance, and license instead of committing the data.

The public snapshot does not contain the external walking teacher or its
numerical observation/action contract. If you have independently verified that
your permission covers rollout generation and model training, record its joint
order and runtime action scale alongside each rollout. Public offline trainers
read that metadata from the rollout instead of assuming a vendor-specific
contract. The teacher used in the historical SONIC-A3 experiments is still
under the rights review documented in `EXTERNAL_TEACHER_RIGHTS.md`.

## 6. Verify source and assets

Pure source checks:

```bash
python tools/verify_a3_source.py
python tools/check_release.py
```

Isaac Lab standing smoke test:

```bash
python scripts/a3/validate_standing.py --seconds 2 --headless
```

Retargeted-motion check, after creating a motion file:

```bash
python scripts/a3/validate_retargeted_motion.py --help
```

Treat GUI teleport playback as a kinematic check only. It does not prove that a
policy can execute the motion under gravity and contact.

## 7. Training entry points

Native A3 Encoder + Kinematic Decoder configuration:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_a3_kin_stage1 \
  num_envs=16 \
  use_wandb=false \
  headless=true \
  ++algo.config.num_learning_iterations=2
```

The command above is only a smoke test. Verify data paths and configuration
snapshots before increasing environments or iterations.

Standalone locomotion entry points:

```bash
python scripts/a3/train_a3_locomotion.py --help
python scripts/a3/play_a3_locomotion.py --help
```

The locomotion task is experimental and is not a released walking model.

### G1-to-A3 alignment tools

The repository includes the alignment pipeline, but not its inputs. You must
provide separately licensed SONIC G1 weights, A3 checkpoints, and paired or
schema-v2 rollout data. Inspect each interface before running a long job:

```bash
python scripts/a3/audit_g1_a3_alignment_pairs.py --help
python scripts/a3/audit_g1_a3_token_baseline.py --help
python scripts/a3/prepare_g1_a3_alignment_feature_dataset.py --help
python scripts/a3/train_g1_a3_latent_adapter.py --help
python scripts/a3/evaluate_g1_a3_adapter_decoder.py --help
```

For broader G1 motions, use the gated BONES-SEED dataset only after accepting
its license on Hugging Face. Keep the archive outside this Git checkout and do
not redistribute it from this repository. A G1 trajectory is not by itself an
alignment pair: generate and validate the corresponding A3 reference before
using it as supervised cross-embodiment data.

## 8. Checkpoint discipline

Never load a checkpoint without its matching:

- action joint order;
- action scale;
- default pose;
- observation order/history;
- control timestep and decimation;
- actuator gains and delay/filter semantics.

Store large models outside ordinary Git. If Git LFS is used later, commit the
`.gitattributes` rules before the first model file and review storage quotas.
