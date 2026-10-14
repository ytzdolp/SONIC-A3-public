# SONIC-A3

An unofficial personal research project adapting NVIDIA's SONIC whole-body
motion representation and control stack to the Zhiyuan A3 humanoid.

> Work in progress. A3 motion representation and the protected Round2 episodic
> locomotion baseline have been validated in Isaac Lab. The corrected MuJoCo
> bridge passes its state/history contract. Round2 completes 52/54 known-command
> and 54/54 held-out-command four-second windows; under the three-second
> reachable multi-initial recovery gate it passes all 108/108 windows.
> Continuous navigation, arbitrary-state recovery, broad whole-body token
> alignment, and full real-robot deployment are not complete. Standing and the
> lowest-speed forward command (`forward_020`, nominally `0.20 m/s`) have passed
> a basic real-robot validation; higher speeds, longer horizons, disturbances,
> recovery, and the remaining motions have not.

> **Evidence and rights boundary:** these are self-reported private-workspace
> results. Private experiments used an A3 walking controller obtained through
> a legitimate project channel as a teacher. This public repository does not
> redistribute that controller, its deployment configuration, numerical
> contract, rollouts, training data, related model weights, or the private
> evaluator needed to reproduce the results. It contains independently written
> integration and training code together with bounded experimental records.
> This project does not claim vendor endorsement. Access to the controller does
> not by itself establish permission to redistribute every derived dataset or
> model; applicable terms must be checked before publication. Never quote
> `108/108` without also stating the four-second
> result (`106/108` overall), the three-second reachable-state threshold, and
> the limited scope of the real-robot validation.

## Relationship to upstream

This repository is derived from
[NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
The NVIDIA project remains the upstream source; A3-specific research changes
are maintained here. This project is not affiliated with or endorsed by
NVIDIA or Zhiyuan Robotics.

- Upstream SONIC source: Apache License 2.0
- Upstream model weights: NVIDIA Open Model License
- A3 robot assets and motion data: not distributed here; obtain and use them
  only under terms granted by their respective owners

The original upstream README is preserved as
[README_UPSTREAM.md](README_UPSTREAM.md).

### Scope compared with upstream SONIC

SONIC-A3 is trained and evaluated on a small, task-specific A3 dataset rather
than the broad motion corpus used to build the upstream SONIC system. The
current A3 implementation supports selected reference-motion reconstruction
and a bounded set of episodic locomotion commands. It is not a general-purpose
whole-body behavior model: arbitrary motions, continuous command navigation,
wide initial-state recovery, and cross-task generalization are not established.

## Current result

The current protected baseline is **A3 episodic locomotion Round2**. It uses a
frozen, correctly configured A3 Dynamic Decoder plus a zero-initialized,
15-joint bounded action adapter trained with two rounds of schema-v2 DAgger.

| Capability | Result |
|---|---|
| A3 Encoder + Kinematic Decoder | 64D token reconstruction independently validated |
| A3 walking controller used as a private teacher | Obtained through a legitimate project channel; no controller, private contract, rollout, or derived model is released |
| Native Dynamic Decoder backbone | Correct-config step8000 selected; 2/6 alone |
| Round2 Isaac Lab known full episodes | 6/6 at seed0; 23/24 across four seeds |
| Round2 Isaac Lab held-out full episodes | 15/24 across four seeds |
| G1-to-A3 token experiment | Offline same-domain fit obtained; strict online test rejected |
| MuJoCo exact-state rollout windows | Four seconds: known 52/54, held-out 54/54; three-second reachable multi-initial gate: 108/108 |
| Continuous navigation / random recovery | Reachable multi-initial 3 s accepted; arbitrary-state recovery not trained |
| Real-robot validation | Basic standing and `forward_020` (nominally `0.20 m/s`) passed; broader speeds, durations, disturbances, recovery, and other motions remain unvalidated |
| Basketball task | Deferred |

The protected Round2 checkpoint is not distributed in this repository. Its
research-workspace SHA-256 is:

```text
9dd32a55fa29fdfbba41cbbcb92c400a019a4899fb4a9993002c67fe352da4ec
```

Round2 depends on teacher-derived training data. Its technical metrics remain
useful as experiment records, while any release of related data or models must
follow the third-party boundary in
[docs/a3/THIRD_PARTY_SOURCE_AND_PUBLICATION_SCOPE.md](docs/a3/THIRD_PARTY_SOURCE_AND_PUBLICATION_SCOPE.md).

### G1-to-A3 token experiment

The current bridge freezes the released SONIC G1 encoder and learns a small
64D latent adapter into the A3 token space. On 10,110 closed-loop validation
states:

| Metric | Raw G1 token | Adapted G1 token |
|---|---:|---:|
| Token cosine vs. native A3 | -0.0566 | 0.9554 |
| Final action cosine vs. native A3 | 0.9966 | 0.99986 |
| Final clipped action RMSE | 0.01997 | 0.00491 |

These metrics do not constitute online acceptance: exact 64D token-window
agreement is 0%, the motion-only OOD token cosine is about 0.066, and the native
A3 path remains slightly closer to the teacher (`0.00898` versus `0.00924` MAE).
A strict Isaac Lab rerun found `forward_040` terminating at frame 529 with
`anchor_pos_xy=0.4402 m`, while the native A3-token control did not terminate
early under the same test. The branch is therefore retained only as an offline
walking-domain alignment experiment. See [docs/a3/RESULTS.md](docs/a3/RESULTS.md).

## What is included

- A3 29-DoF schema and Isaac Lab/MuJoCo order mappings
- Native A3 encoder, kinematic decoder, and dynamic-decoder configurations
- Native Dynamic Decoder, rollout-schema, action-adapter, and evaluation tools
- G1/A3 pair audits, latent-alignment training, decoder evaluation, and runtime bridge
- Retargeting, virtual-motion, reconstruction, and rollout scripts
- Experimental standalone A3 locomotion environment
- Project status and handoff-safe technical documentation

Not included:

- A3 URDF, meshes, or MJCF; obtain the public robot model separately from
  [AgibotTech/A3-A3U-robot-model](https://github.com/AgibotTech/A3-A3U-robot-model)
- BONES-SEED, basketball, external-policy, or locally collected training data
- any external deployment source, numerical walking contract, ONNX model, or telemetry
- NVIDIA or locally trained checkpoints
- experiment logs, virtual environments, internal dialogs, or credentials
- unchanged G1 deployment/teleoperation binaries available from upstream

## Repository layout

```text
SONIC-A3/
├── gear_sonic/                  # SONIC source plus A3 integration
├── a3_locomotion/               # experimental standalone A3 locomotion task
├── scripts/a3/                  # data, training, DAgger, alignment, and validation tools
├── docs/a3/                     # installation, results, assets, and release notes
├── SONIC_A3_PROJECT_INTRODUCTION.md
├── SONIC_A3_PROJECT_PROGRESS.md
├── LICENSE
└── README_UPSTREAM.md
```

## Installation

See [docs/a3/INSTALLATION.md](docs/a3/INSTALLATION.md). In short:

1. install Isaac Sim and Isaac Lab 2.3+;
2. install `gear_sonic[training]` in the Isaac Lab Python environment;
3. place legally obtained A3 assets at the documented paths;
4. download required upstream model weights separately;
5. run the source and environment checks before training or evaluation.

This source-only check does not require Isaac Lab or robot assets:

```bash
python tools/verify_a3_source.py
python tools/check_release.py
```

After Isaac Lab and A3 assets are installed:

```bash
python scripts/a3/validate_standing.py --seconds 2 --headless
```

Do not start long training runs until the checks report the intended SI
actuator gains, stable default pose, 29-joint order, observation layout, action
scale, and checkpoint hashes.

## Development path

The accepted path is:

```text
A3 motion and 29-DoF contract
        -> native Encoder + Kinematic Decoder
        -> privately obtained A3 walking teacher (no redistribution)
        -> correct-config native Dynamic Decoder (step8000)
        -> schema-v2 student-state DAgger
        -> bounded action adapter Round2
```

The corrected private-workspace MuJoCo evaluator restores exact robot/history
state and recomputes state-dependent observations. On known commands Round2
passes 52/54 four-second windows; both failures occur after 3.6 s. A held-out
manifest covering speed interpolation, slow turns, and curved walking passes
54/54 four-second windows. The combined three-second reachable multi-initial
gate is therefore 108/108. This uses physically reachable rollout states, not
arbitrary joint randomization. The contract-specific bridge, assets, models,
and rollout files are not distributed by this public repository.

Detailed history and current decisions are in
[SONIC_A3_PROJECT_PROGRESS.md](SONIC_A3_PROJECT_PROGRESS.md).

## Models, data, and publication

This repository intentionally contains no checkpoints, private motion data,
external walking model, deployment package, or numerical walking-policy
contract. Offline training utilities may consume an action contract recorded
in a user-provided rollout only after the user verifies the applicable rights,
instead of embedding that contract in public source. See
[docs/a3/ASSETS_AND_LICENSES.md](docs/a3/ASSETS_AND_LICENSES.md).
When publishing models separately, record at least:

- source commit and upstream revision;
- exact environment and actuator configuration;
- joint order, observation layout, action scale, and update frequency;
- dataset manifest, split policy, and hashes;
- source, backbone, adapter, and runtime-contract hashes;
- simulator and Isaac Lab versions;
- applicable model, robot-asset, and dataset licenses.

## Publication follow-up

See [the remaining evidence and hosting checklist](docs/a3/PUBLICATION_FOLLOWUP.md)
for third-party publication scope, evaluation records, and previously
published objects.

## License and attribution

Retain [LICENSE](LICENSE), upstream copyright headers, and third-party notices.
See [docs/a3/ASSETS_AND_LICENSES.md](docs/a3/ASSETS_AND_LICENSES.md) before adding
robot assets, datasets, or model weights.
