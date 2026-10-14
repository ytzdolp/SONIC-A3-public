"""AgiBot A3 configuration for SONIC whole-body motion tracking.

This project-authored Isaac Lab configuration targets the public AgiBot A3
robot-model contract. The actuator gains below use Isaac Lab's user-facing SI
units: N·m/rad for stiffness and N·m·s/rad for damping. Robot assets are
installed separately under their own license.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = "gear_sonic/data/assets"

# ---------------------------------------------------------------------------
# Armature constants used by this project's A3 simulation configuration.
# Validate them against the selected public A3 model revision before training.
# ---------------------------------------------------------------------------
ARMATURE_PFP_78_58 = 30.118e-6 * 20.03 * 20.03
ARMATURE_PFP_93_65 = 138.5069e-6 * 21.906 * 21.906
ARMATURE_PFP_41_48 = 1.359e-6 * 24.415 * 24.415
ARMATURE_PFP_59_60 = 10.1374e-6 * 22.136 * 22.136
ARMATURE_PFP_110_75 = 300.851e-6 * 20 * 20

# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------
A3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ASSET_DIR}/robot_description/urdf/a3/urdf/a3.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        collider_type="convex_decomposition",
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        self_collision=False,
        make_instanceable=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            drive_type="force",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.063),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Legs.
            "left_hip_pitch_joint": -0.1311,
            "right_hip_pitch_joint": -0.1311,
            "left_hip_roll_joint": 0.0056,
            "right_hip_roll_joint": -0.0056,
            "left_hip_yaw_joint": -0.0348,
            "right_hip_yaw_joint": 0.0348,
            "left_knee_joint": 0.2468,
            "right_knee_joint": 0.2468,
            "left_ankle_pitch_joint": -0.1204,
            "right_ankle_pitch_joint": -0.1204,
            "left_ankle_roll_joint": -0.0078,
            "right_ankle_roll_joint": 0.0078,
            # Waist.
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # Arms (shoulders + elbows + wrists).
            "left_shoulder_pitch_joint": 0.3,
            "right_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.12,
            "right_shoulder_roll_joint": -0.12,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.8,
            "right_elbow_joint": 0.8,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # -------------------------------------------------------------------
        # Legs – project simulation parameters for hip and knee joints.
        # -------------------------------------------------------------------
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": 220.0,
                ".*_hip_yaw_joint": 220.0,
                ".*_hip_pitch_joint": 220.0,
                ".*_knee_joint": 320.0,
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": 12.042771838760874,
                ".*_hip_yaw_joint": 12.042771838760874,
                ".*_hip_pitch_joint": 12.042771838760874,
                ".*_knee_joint": 14.660765716752367,
            },
            stiffness={
                ".*_hip_roll_joint": 120,
                ".*_hip_yaw_joint": 80,
                ".*_hip_pitch_joint": 80,
                ".*_knee_joint": 250.0,
            },
            damping={
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 3.0,
                ".*_hip_pitch_joint": 3.0,
                ".*_knee_joint": 8.0,
            },
            armature={
                ".*_hip_roll_joint": ARMATURE_PFP_93_65,
                ".*_hip_yaw_joint": ARMATURE_PFP_93_65,
                ".*_hip_pitch_joint": ARMATURE_PFP_93_65,
                ".*_knee_joint": ARMATURE_PFP_110_75,
            },
            friction={
                ".*_hip_roll_joint": 0.15,
                ".*_hip_yaw_joint": 0.15,
                ".*_hip_pitch_joint": 0.15,
                ".*_knee_joint": 0.15,
            },
            dynamic_friction={
                ".*_hip_roll_joint": 0.1,
                ".*_hip_yaw_joint": 0.1,
                ".*_hip_pitch_joint": 0.1,
                ".*_knee_joint": 0.1,
            },
        ),
        # -------------------------------------------------------------------
        # Feet – project simulation parameters for ankle joints.
        # -------------------------------------------------------------------
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 118.2,
                ".*_ankle_roll_joint": 54.75,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 10.8,
                ".*_ankle_roll_joint": 19.37,
            },
            stiffness={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            armature={
                ".*_ankle_pitch_joint": ARMATURE_PFP_78_58 * 5.333,
                ".*_ankle_roll_joint": ARMATURE_PFP_78_58 * 1.66562,
            },
            friction={
                ".*_ankle_pitch_joint": 0.25,
                ".*_ankle_roll_joint": 0.25,
            },
            dynamic_friction={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.2,
            },
        ),
        # -------------------------------------------------------------------
        # Waist – project simulation parameters.
        # -------------------------------------------------------------------
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 220.0,
                "waist_roll_joint": 46.0,
                "waist_pitch_joint": 115,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 12.042771838760874,
                "waist_roll_joint": 22.7,
                "waist_pitch_joint": 9.24785,
            },
            stiffness={
                "waist_yaw_joint": 85,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            damping={
                "waist_yaw_joint": 3.0,
                "waist_roll_joint": 2.0,
                "waist_pitch_joint": 2.0,
            },
            armature={
                "waist_yaw_joint": ARMATURE_PFP_93_65,
                "waist_roll_joint": ARMATURE_PFP_78_58 * 1.21,
                "waist_pitch_joint": ARMATURE_PFP_78_58 * 7.3,
            },
            friction={
                "waist_yaw_joint": 0.15,
                "waist_roll_joint": 0.15,
                "waist_pitch_joint": 0.15,
            },
            dynamic_friction={
                "waist_yaw_joint": 0.1,
                "waist_roll_joint": 0.1,
                "waist_pitch_joint": 0.1,
            },
        ),
        # -------------------------------------------------------------------
        # Arms – shoulders, elbows, and wrists.
        # -------------------------------------------------------------------
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_roll_joint": 24.0,
                ".*_wrist_pitch_joint": 6.0,
                ".*_wrist_yaw_joint": 6.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.613568165555769,
                ".*_shoulder_roll_joint": 13.613568165555769,
                ".*_shoulder_yaw_joint": 14.660765716752367,  # PFP-59-60 rated 140 rpm (doc 0716 §5)
                ".*_elbow_joint": 14.660765716752367,  # PFP-59-60 rated 140 rpm
                ".*_wrist_roll_joint": 14.660765716752367,  # PFP-59-60 rated 140 rpm
                ".*_wrist_pitch_joint": 12.775810124598491,
                ".*_wrist_yaw_joint": 12.775810124598491,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 30.0,
                ".*_elbow_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
                ".*_wrist_yaw_joint": 2.0,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_PFP_78_58,
                ".*_shoulder_roll_joint": ARMATURE_PFP_78_58,
                ".*_shoulder_yaw_joint": ARMATURE_PFP_59_60,
                ".*_elbow_joint": ARMATURE_PFP_59_60,
                ".*_wrist_roll_joint": ARMATURE_PFP_59_60,
                ".*_wrist_pitch_joint": ARMATURE_PFP_41_48,
                ".*_wrist_yaw_joint": ARMATURE_PFP_41_48,
            },
            friction={
                ".*_shoulder_pitch_joint": 0.1,
                ".*_shoulder_roll_joint": 0.1,
                ".*_shoulder_yaw_joint": 0.1,
                ".*_elbow_joint": 0.1,
                ".*_wrist_roll_joint": 0.1,
                ".*_wrist_pitch_joint": 0.1,
                ".*_wrist_yaw_joint": 0.1,
            },
            dynamic_friction={
                ".*_shoulder_pitch_joint": 0.1,
                ".*_shoulder_roll_joint": 0.1,
                ".*_shoulder_yaw_joint": 0.1,
                ".*_elbow_joint": 0.1,
                ".*_wrist_roll_joint": 0.1,
                ".*_wrist_pitch_joint": 0.1,
                ".*_wrist_yaw_joint": 0.1,
            },
        ),
    },
)

# ---------------------------------------------------------------------------
# Isaac Lab ↔ MuJoCo mapping tables validated for this project's A3 model.
# ---------------------------------------------------------------------------
A3_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]

A3_ISAACLAB_TO_MUJOCO_DOF = [
    2,  5,  8,  11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28, 0,  3,  6,
    9,  13, 17, 1,  4,  7,  10, 14, 18,
]

A3_MUJOCO_TO_ISAACLAB_DOF = [
    17, 23, 0, 18, 24, 1, 19, 25, 2, 20,
    26, 3, 10, 21, 27, 4, 11, 22, 28, 5,
    12, 6, 13, 7, 14, 8, 15, 9, 16,
]

# Canonical semantic schema used by the A3 tokenizer and action decoder.  The
# motion library stores DOFs in MuJoCo order while Isaac Lab exposes the
# articulation in its traversal order.  Keeping both name lists next to the
# permutations prevents a coincidental 29-DOF match from being mistaken for
# semantic compatibility with G1.
A3_MUJOCO_DOF_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

A3_ISAACLAB_DOF_NAMES = [
    A3_MUJOCO_DOF_NAMES[i] for i in A3_MUJOCO_TO_ISAACLAB_DOF
]


def validate_a3_joint_schema() -> None:
    """Fail fast when an A3 asset/order edit breaks the 29-DOF contract."""
    count = len(A3_MUJOCO_DOF_NAMES)
    assert count == 29
    assert len(set(A3_MUJOCO_DOF_NAMES)) == count
    assert sorted(A3_ISAACLAB_TO_MUJOCO_DOF) == list(range(count))
    assert sorted(A3_MUJOCO_TO_ISAACLAB_DOF) == list(range(count))
    assert [A3_ISAACLAB_TO_MUJOCO_DOF[i] for i in A3_MUJOCO_TO_ISAACLAB_DOF] == list(
        range(count)
    )


validate_a3_joint_schema()

A3_ISAACLAB_TO_MUJOCO_BODY = [
    0,  3,  6,  9,  12, 16, 20, 22, 24, 26,
    28, 13, 17, 21, 23, 25, 27, 29, 1,  4,
    7,  10, 14, 18, 2,  5,  8,  11, 15, 19,
]

A3_MUJOCO_TO_ISAACLAB_BODY = [
    0, 18, 24, 1, 19, 25, 2, 20, 26, 3,
    21, 27, 4, 11, 22, 28, 5, 12, 23, 29,
    6, 13, 7, 14, 8, 15, 9, 16, 10, 17,
]

A3_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": A3_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": A3_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": A3_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": A3_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": A3_MUJOCO_TO_ISAACLAB_BODY,
}


# Position-target range follows the convention used by the existing SONIC
# embodiments: one quarter of the static effort/stiffness displacement.
A3_ACTION_SCALE = {}
for actuator in A3_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = dict.fromkeys(names, effort)
    if not isinstance(stiffness, dict):
        stiffness = dict.fromkeys(names, stiffness)
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            A3_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
