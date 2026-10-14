"""Pure-Python checks for the publishable SONIC-A3 source tree."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
A3_ROBOT = ROOT / "gear_sonic/envs/manager_env/robots/a3.py"
REQUIRED = (
    A3_ROBOT,
    ROOT / "a3_locomotion/env_cfg.py",
    ROOT / "gear_sonic/config/actor_critic/encoders/a3_mf_mlp.yaml",
    ROOT / "gear_sonic/config/actor_critic/decoders/a3_kin_mf_mlp.yaml",
    ROOT / "gear_sonic/config/actor_critic/decoders/a3_dyn_mlp.yaml",
    ROOT / "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3_kin_stage1.yaml",
    ROOT / "gear_sonic/trl/modules/g1_a3_runtime_adapter.py",
    ROOT / "scripts/a3/audit_g1_a3_alignment_pairs.py",
    ROOT / "scripts/a3/prepare_g1_a3_alignment_feature_dataset.py",
    ROOT / "scripts/a3/train_g1_a3_latent_adapter.py",
    ROOT / "scripts/a3/evaluate_g1_a3_adapter_decoder.py",
    ROOT / "docs/a3/RESULTS.md",
)


def literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise KeyError(name)


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required A3 source files: {missing}")
    tree = ast.parse(A3_ROBOT.read_text(encoding="utf-8"), filename=str(A3_ROBOT))
    forward = literal_assignment(tree, "A3_ISAACLAB_TO_MUJOCO_DOF")
    inverse = literal_assignment(tree, "A3_MUJOCO_TO_ISAACLAB_DOF")
    if sorted(forward) != list(range(29)) or sorted(inverse) != list(range(29)):
        raise ValueError("A3 DoF mapping is not a 29-element permutation")
    if [forward[index] for index in inverse] != list(range(29)):
        raise ValueError("A3 forward/inverse DoF mappings are inconsistent")
    text = A3_ROBOT.read_text(encoding="utf-8")
    forbidden = ("PD_GAIN_DEG_TO_RAD", "180.0 / math.pi")
    found = [token for token in forbidden if token in text]
    if found:
        raise ValueError(f"forbidden duplicate PD conversion found: {found}")
    print("SONIC_A3_SOURCE_OK dofs=29 mapping=invertible duplicate_pd_conversion=absent")


if __name__ == "__main__":
    main()
