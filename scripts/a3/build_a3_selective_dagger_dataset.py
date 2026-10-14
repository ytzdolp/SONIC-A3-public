#!/usr/bin/env python3
"""Build a conservative schema-v2 A3 DAgger dataset.

Teacher labels are retained only for explicitly selected motion keys.  Other
rows use the recorded source-student action as a behavior-preservation target.
All runtime adapter inputs and episode boundaries remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-motion", action="append", default=[])
    args = parser.parse_args()
    if not args.teacher_motion:
        parser.error("at least one --teacher-motion is required")
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must differ")

    with np.load(args.source, allow_pickle=False) as source:
        if "schema_json" not in source.files:
            raise RuntimeError(f"{args.source}: missing schema_json")
        arrays = {key: source[key].copy() for key in source.files}

    metadata = json.loads(str(arrays["schema_json"].item()))
    if metadata.get("schema_version") != 2:
        raise RuntimeError("selective DAgger requires schema-v2 adapter data")
    motion_keys = metadata.get("motion_keys")
    if not isinstance(motion_keys, list) or not motion_keys:
        raise RuntimeError("source metadata has no motion_keys")
    requested = set(args.teacher_motion)
    missing = requested - set(motion_keys)
    if missing:
        raise RuntimeError(f"teacher motions absent from source: {sorted(missing)}")

    unique_motion_id = arrays["unique_motion_id"].astype(np.int64, copy=False)
    row_motion_keys = np.asarray(motion_keys, dtype=str)[unique_motion_id]
    teacher_mask = np.isin(row_motion_keys, sorted(requested))
    original_teacher_targets = arrays["targets"].copy()
    arrays["targets"][~teacher_mask] = arrays["raw_student_actions"][~teacher_mask]

    selected_indices = np.asarray(metadata["adapter_joint_indices"], dtype=np.int64)
    teacher_delta = np.abs(
        original_teacher_targets[:, selected_indices]
        - arrays["raw_student_actions"][:, selected_indices]
    )
    metadata["supervision_mode"] = "selective_dagger_with_behavior_preservation"
    metadata["source_dataset"] = str(args.source.resolve())
    metadata["source_dataset_sha256"] = file_sha256(args.source)
    metadata["teacher_target_motion_keys"] = sorted(requested)
    metadata["teacher_labeled_samples"] = int(np.count_nonzero(teacher_mask))
    metadata["behavior_preservation_samples"] = int(np.count_nonzero(~teacher_mask))
    metadata["original_teacher_student_mae_selected_joints"] = float(teacher_delta.mean())
    metadata["effective_teacher_student_mae_selected_joints"] = float(
        teacher_delta[teacher_mask].sum() / (len(selected_indices) * np.count_nonzero(teacher_mask))
    )
    arrays["schema_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        f"A3_SELECTIVE_DAGGER_SAVED path={args.output} "
        f"teacher_samples={metadata['teacher_labeled_samples']} "
        f"preserve_samples={metadata['behavior_preservation_samples']} "
        f"teacher_motions={sorted(requested)}"
    )


if __name__ == "__main__":
    main()
