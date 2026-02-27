import pytest
import numpy as np
from magbotsim.utils import rotations_utils

# The following tests are developed using Gemini.

SQRT2_2 = np.sqrt(2) / 2


@pytest.mark.parametrize(
    'euler_in, expected_quat',
    [
        # 1. Identity: No rotation
        (np.array([[0, 0, 0]]), np.array([[1, 0, 0, 0]])),
        # 2. 90 deg rotation around X (Roll)
        # q = [cos(45), sin(45), 0, 0]
        (np.array([[np.pi / 2, 0, 0]]), np.array([[SQRT2_2, SQRT2_2, 0, 0]])),
        # 3. 90 deg rotation around Y (Pitch)
        # q = [cos(45), 0, sin(45), 0]
        (np.array([[0, np.pi / 2, 0]]), np.array([[SQRT2_2, 0, SQRT2_2, 0]])),
        # 4. 90 deg rotation around Z (Yaw)
        # q = [cos(45), 0, 0, sin(45)]
        (np.array([[0, 0, np.pi / 2]]), np.array([[SQRT2_2, 0, 0, SQRT2_2]])),
    ],
)
def test_euler2quat_values(euler_in, expected_quat):
    res = rotations_utils.vec_euler2quat(euler_in)
    assert np.allclose(res, expected_quat)
    assert np.allclose(np.linalg.norm(res, axis=1), 1.0)


@pytest.mark.parametrize(
    'quat_in, expected_euler',
    [
        # identity
        (np.array([[1, 0, 0, 0]]), np.array([[0, 0, 0]])),
        # 90 deg X
        (np.array([[SQRT2_2, SQRT2_2, 0, 0]]), np.array([[np.pi / 2, 0, 0]])),
        # 90 deg Z
        (np.array([[SQRT2_2, 0, 0, SQRT2_2]]), np.array([[0, 0, np.pi / 2]])),
    ],
)
def test_quat2euler_values(quat_in, expected_euler):
    res = rotations_utils.vec_quat2euler(quat_in)
    assert np.allclose(res, expected_euler)


def test_roundtrip_euler_quat_euler():
    """Verify that E -> Q -> E returns the original angles."""
    # create random angles between -pi/4 and pi/4 to avoid gimbal lock regions for this simple test
    random_euler = (np.random.rand(10, 3) - 0.5) * (np.pi / 2)
    quats = rotations_utils.vec_euler2quat(random_euler)
    reconstructed_euler = rotations_utils.vec_quat2euler(quats)
    assert np.allclose(random_euler, reconstructed_euler)


@pytest.mark.parametrize(
    'mat_in, expected_quat',
    [
        # Identity Matrix
        (np.eye(3).reshape(1, 3, 3), np.array([[1, 0, 0, 0]])),
        # 180 deg around X: [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        # triggers mask1 (R[0,0] is largest)
        (np.array([[[1, 0, 0], [0, -1, 0], [0, 0, -1]]]), np.array([[0, 1, 0, 0]])),
        # 180 deg around Z: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        # triggers mask3 (R[2,2] is largest)
        (np.array([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]), np.array([[0, 0, 0, 1]])),
    ],
)
def test_mat2quat_special_cases(mat_in, expected_quat):
    res = rotations_utils.vec_mat2quat(mat_in)
    # quaternions q and -q represent the same rotation
    # check if absolute values match or if the dot product is ~1
    for i in range(len(res)):
        dot_product = np.abs(np.dot(res[i], expected_quat[i]))
        assert np.isclose(dot_product, 1.0)


def test_gimbal_lock_pitch():
    """Test behavior at the pitch singularity (+/- 90 degrees)."""
    # Pitch at 90 degrees
    euler = np.array([[0, np.pi / 2, 0]])
    quat = rotations_utils.vec_euler2quat(euler)
    res_euler = rotations_utils.vec_quat2euler(quat)
    # in gimbal lock, individual Roll/Yaw are ambiguous, but their sum/diff is not.
    # check if the reconstructed pitch is at least correct.
    assert np.isclose(res_euler[0, 1], np.pi / 2)
