import msgpack
import msgpack_numpy as msgpack_numpy
import numpy as np

from gear_sonic.utils.teleop.input_readers import (
    build_body_pose_sample,
    decode_msgpack_byte_multi_array,
)


def test_decode_msgpack_byte_multi_array_from_byte_chunks():
    payload = {
        "timestamp": 123456789,
        "joint_positions": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "joint_orientations": [[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
    }
    packed = msgpack.packb(payload, default=msgpack_numpy.encode, use_bin_type=True)
    byte_chunks = [bytes([value]) for value in packed]

    decoded = decode_msgpack_byte_multi_array(
        byte_chunks,
        msgpack_module=msgpack,
        msgpack_numpy_module=msgpack_numpy,
    )

    assert decoded["timestamp"] == payload["timestamp"]
    assert decoded["joint_positions"] == payload["joint_positions"]
    assert decoded["joint_orientations"] == payload["joint_orientations"]


def test_build_body_pose_sample_uses_existing_teleop_shape():
    payload = {
        "timestamp": 1_000_000_100,
        "joint_positions": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "joint_orientations": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    }

    sample, stamp_ns, fps_ema = build_body_pose_sample(
        payload,
        prev_stamp_ns=1_000_000_000,
        fps_ema=0.0,
    )

    assert sample is not None
    assert stamp_ns == payload["timestamp"]
    assert sample["body_poses_np"].shape == (24, 7)
    np.testing.assert_allclose(sample["body_poses_np"][0], np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(sample["body_poses_np"][1], np.array([0.4, 0.5, 0.6, 0.0, 0.0, 1.0, 0.0]))
    assert sample["dt"] == 1e-7
    assert fps_ema == 1.0 / 1e-7
