# Assets, data, models, and licenses

## Repository code

SONIC source code is derived from NVlabs/GR00T-WholeBodyControl and distributed
under the source-code terms contained in the repository `LICENSE`. Preserve all
upstream copyright headers and attribution.

This repository is unofficial and must not imply endorsement by NVIDIA or
Zhiyuan Robotics.

## NVIDIA model weights

Upstream SONIC weights are governed separately by the NVIDIA Open Model License
contained in `LICENSE`. Do not assume the Apache-2.0 source-code terms apply to
weights. Review redistribution and attribution requirements before publishing
an original or derivative checkpoint.

## A3 robot assets

The public AgiBot A3/A3U robot models are available separately from
`AgibotTech/A3-A3U-robot-model` under the Mulan Permissive Software License v2.
This repository does not copy those URDF, MJCF, mesh, or CAD files. Pin the
model revision used by an experiment and preserve its license and notices if
you redistribute model files.

For every asset considered for publication, record:

- owner and source URL or delivery channel;
- exact license or written permission;
- whether modification and redistribution are allowed;
- attribution requirements;
- whether public cloud hosting is allowed.

Private GitHub visibility does not override these terms.

## Privately obtained A3 walking controller

The project owner states that the private research workspace obtained an A3
walking controller through a legitimate project channel and used it as a
teacher. This public snapshot excludes that controller, its deployment program,
original configuration, recorded rollouts, exact observation/action contract,
contract-specific runtime bridge, and related model weights. The project does
not claim vendor endorsement.

The scope for public release of teacher-derived data or models has **not** been
documented in this repository. Until applicable terms are confirmed, this
project must not claim that access covers public distribution of rollouts or
derived checkpoints. Absence of the controller weights from Git is not evidence
that those additional publication rights have been granted.

The public offline trainers accept runtime scale and joint metadata from a
user-provided rollout only when that user has independently verified the
applicable rights. This preserves the general Dynamic Decoder, DAgger-dataset,
and adapter methods without publishing an external policy contract. Do not
describe these project-authored methods as vendor code.

See
[THIRD_PARTY_SOURCE_AND_PUBLICATION_SCOPE.md](THIRD_PARTY_SOURCE_AND_PUBLICATION_SCOPE.md)
for the recorded source and public-release boundary.

## Third-party source code retained in the tree

The body-model utilities include modifications derived from HuMoR, Copyright
(c) 2021 Davis Rempe, under the MIT License. Rotation-conversion utilities
include code derived from PyTorch3D, Copyright (c) Meta Platforms, Inc. and
affiliates, under the BSD 3-Clause License. Their complete notices are retained
in:

- `legal/MIT License - HuMoR.txt`
- `legal/BSD License - PyTorch3D.txt`

An unused `smpl_to_openpose` implementation copied from SMPLify-X was removed
from the current tree because permission to redistribute that code was not
established. This removal does not by itself remove the blob from older Git
commits; publishing a cleaned repository therefore requires the history review
described in `RELEASE_CHECKLIST.md`.

## Motion and basketball data

Human motion, teleoperation recordings, real-robot telemetry, and basketball
mission data may include third-party copyright, confidentiality, or personal
data. Keep them outside Git unless provenance and permission are explicit.

Recommended manifest fields:

```text
filename, sha256, source, owner, license, fps, joint schema, review status
```

## Locally trained checkpoints

A locally trained checkpoint may still derive from an upstream checkpoint or
licensed dataset. Before distribution, document its full lineage. Private model
storage is preferred while that lineage is unresolved.
