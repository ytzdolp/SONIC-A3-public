# SONIC-A3 Results Snapshot

Updated: 2026-10-14

This document records accepted research results and their limits. Checkpoints,
robot assets, rollout datasets, and private motion data are intentionally not
distributed in this repository.

These values are self-reported from the private workspace. The public snapshot
does not include the relevant weights, raw logs, rollout manifests, or the
contract-specific MuJoCo evaluator, so a third party cannot independently
reproduce the tables from this repository alone. Private experiments used an
A3 walking controller obtained through a legitimate project channel as a
teacher. The controller, deployment configuration, numerical contract,
rollouts, training data, and related weights are not redistributed, and the
project does not claim vendor endorsement. Access to the controller does not by
itself establish redistribution permission for every derived artifact.

## Acceptance policy

A model is not accepted because training loss decreases or a reference moves in
the GUI. The project checks, in order:

1. joint, observation, action, frame-rate, and checkpoint contracts;
2. offline token and action reconstruction on trajectory-level holdouts;
3. deterministic closed-loop Isaac Lab episodes;
4. multi-seed and held-out commands;
5. visual inspection only after the numerical checks pass.

## A3 episodic locomotion Round2

Architecture:

```text
frozen A3 Encoder + FSQ token
              +
930D term-major proprioception history
              -> frozen correct-config step8000 Dynamic Decoder
              -> bounded 15-joint action adapter
              -> 29D normalized A3 action
```

The adapter is zero-initialized, bounded to an absolute residual of 0.25, and
trained with motion-balanced behavior cloning over two rounds of schema-v2
student-state DAgger. The protected research checkpoint has SHA-256:

```text
9dd32a55fa29fdfbba41cbbcb92c400a019a4899fb4a9993002c67fe352da4ec
```

Accepted closed-loop results:

| Evaluation | Completed episodes |
|---|---:|
| Six known motions, seed0 | 6/6 |
| Six known motions, four seeds | 23/24 |
| Six held-out motions, four seeds | 15/24 |
| Default observation noise | 20/24 |
| Observation noise disabled | 21/24 |
| Double observation noise | 16/24 |
| Mild push | 14/24 |
| Combined dynamics randomization | 11/24 |

Round2 is accepted as reset-per-episode reference tracking. Its bounded
MuJoCo multi-initial result is recorded below. It is not a continuous
command-navigation, arbitrary-state-recovery, or broadly validated sim-to-real
policy. Basic standing and `forward_020` have passed limited real-robot
validation; broader conditions remain unvalidated.

## Limited real-robot validation

Standing and the lowest-speed `forward_020` command (nominally `0.20 m/s`) have
passed a basic real-robot validation. This result is self-reported from the
private workspace and does not cover higher speeds, long-duration stability,
disturbances, recovery, continuous navigation, or the remaining motions.

## G1-to-A3 walking-domain alignment v2

The alignment experiment freezes the released SONIC G1 Encoder, both A3
decoders, FSQ, and Round2. Only a `64 -> 128 -> 64` latent adapter (16,576
parameters) is optimized.

Dataset contract:

- input: 10 future frames × (58 joint features + 6 root-orientation features);
- encoder input: 640D;
- token: 2 × 32D FSQ;
- train: 37,135 closed-loop states from 9 trajectories;
- validation: 10,110 states from 6 independent trajectories;
- joint mapping, source checkpoint, backbone, and runtime action contract are
  verified before training.

Validation results:

| Metric | Raw frozen G1 | G1 + latent adapter |
|---|---:|---:|
| Token cosine vs. native A3 | -0.05660 | 0.95537 |
| Token RMSE vs. native A3 | 0.46655 | 0.12914 |
| Base decoder action RMSE | 0.02915 | 0.00707 |
| Final action RMSE | 0.02012 | 0.00494 |
| Final clipped action RMSE | 0.01997 | 0.00491 |
| Final clipped action cosine | 0.99666 | 0.999857 |
| Target MAE | 0.01406 | 0.00924 |

The native A3 path has target MAE `0.00898`; the adapted path is therefore
close but not identical to the native encoder.

The previous online claim was invalid because the evaluator continued after an
individual environment reset and reported only the global 650-step cap. A strict
rerun recorded `forward_040` terminating at frame 529 with
`anchor_pos_xy=0.4402 m`; the native A3-token path did not terminate early under
the same conditions. This branch is retained as an offline walking-domain
representation experiment and has not passed online control acceptance.

### Known limitation

An earlier ideal-reference adapter generalized well to held-out motion-only
windows but failed on real closed-loop states. The v2 adapter fixes the
closed-loop distribution, while idealized motion-only input remains out of
distribution (token cosine approximately 0.066). The supported claim is
therefore offline walking-runtime alignment, not an accepted online bridge or a
universal G1/A3 token space.

## Accepted with bounded scope: MuJoCo sim-to-sim

After correcting the private-workspace state/history bridge, the protected
Round2 and the private teacher positive control were run from identical
rollout-derived joint, root, history, phase, and previous-action states. The
public repository does not distribute that private controller contract, the robot
assets, checkpoints, rollouts, or the contract-specific bridge.

Four-second diagnostic results:

| Manifest | Protected Round2 | Private teacher positive control |
|---|---:|---:|
| Six known commands | 52/54 | 54/54 |
| Six held-out commands | 54/54 | 54/54 |

The held-out commands cover forward 0.25/0.35 m/s, slow turns at +/-0.15 rad/s,
and curved walking at 0.30 m/s with +/-0.15 rad/s yaw. Its schema-v2 source
audit checked 10,941 frames and 10,923 consecutive transitions; previous-action
error was zero and the largest history-state error was about `7.4e-7`.

The two known-command failures occur at 3.64 s and 3.70 s, so the primary
four-second diagnostic is **106/108** overall. Under the separate three-second
reachable multi-initial gate, Round2 passes **108/108**
windows across both manifests. The starts are physically reached rollout states,
not arbitrary random poses. This acceptance does not establish continuous
navigation, long-horizon recovery, or broader sim-to-real readiness beyond the
limited standing and `forward_020` real-robot validation above.

Do not report the three-second result as an unqualified "100% success rate."
Any summary must include the four-second result, bounded initial-state and time
horizon, unavailable public reproduction artifacts, and the limited scope of
the real-robot validation.

## Reproduction entry points

The source-only stages are implemented by:

- `scripts/a3/audit_g1_a3_alignment_pairs.py`
- `scripts/a3/audit_g1_a3_token_baseline.py`
- `scripts/a3/prepare_g1_a3_alignment_dataset.py`
- `scripts/a3/prepare_g1_a3_alignment_feature_dataset.py`
- `scripts/a3/train_g1_a3_latent_adapter.py`
- `scripts/a3/evaluate_g1_a3_adapter_decoder.py`
- `gear_sonic/trl/modules/g1_a3_runtime_adapter.py`

The commands require separately obtained G1 model weights, A3 assets,
checkpoints, and rollout data. Do not substitute files solely because their
tensor dimensions match; use the manifest hashes and joint-name audits.
