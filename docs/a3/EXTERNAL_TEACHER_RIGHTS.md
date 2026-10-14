# External teacher rights status

Updated: 2026-10-14

## Current status

The project owner states that the private research workspace obtained an A3
walking controller through a legitimate project channel and used it as a
teacher. The public repository does not redistribute that controller, its
deployment configuration, numerical contract, rollouts, telemetry, training
data, or related model weights.

The repository does not currently contain written terms that establish the
scope for public release of teacher-derived data or models. Accordingly,
SONIC-A3 does **not** represent vendor endorsement or claim that access to the
controller authorizes every possible downstream publication or redistribution.

This status does not establish infringement. It records that the public
release cannot yet verify the relevant rights.

## Public description

Use the following bounded description in public project materials:

> Private experiments used an A3 walking controller obtained through a
> legitimate project channel as a teacher. The public repository does not
> redistribute the controller, deployment configuration, numerical contract,
> rollouts, training data, or related model weights. It contains independently
> written integration and training code together with bounded experimental
> records. The project does not represent vendor endorsement.

Do not shorten this to "officially authorized teacher" or "official walking
policy" unless the provider or vendor provenance and the relevant permission
scope can be documented.

## Confirmation required from the provider

Obtain a written answer covering each item separately:

- use of the actor to generate simulation rollouts and labels;
- training or distilling a Dynamic Decoder, adapter, or other student policy;
- publication of source code that interoperates with the actor;
- publication of performance tables, plots, videos, and qualitative results;
- redistribution of rollouts, telemetry, configuration, or interface details;
- redistribution or public release of derived checkpoints;
- research, commercial, and real-robot deployment scope;
- required attribution, confidentiality terms, and revocation conditions.

Record the provider, date, exact artifact/version, granted uses, excluded uses,
and a durable reference to the permission. Do not commit confidential
correspondence; retain a reviewable internal record.

## Restrictions until confirmation

- Do not publish the external actor, its deployment package, numerical
  contract, rollouts, telemetry, or teacher-derived checkpoints.
- Treat Round2 and other teacher-derived metrics as provisional historical
  experiment records, not as independently reproducible release claims.
- Do not state or imply that the actor, data, results, or derived models are
  provider-approved or vendor-endorsed.
- Do not start new public-release-oriented distillation or data generation
  with the actor solely on the assumption that private access is sufficient.

After confirmation, update this file and the release checklist with the exact
scope. A generic statement such as "permission granted" is not enough.
