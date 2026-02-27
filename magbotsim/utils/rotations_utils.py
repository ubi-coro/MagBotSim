import numpy as np

# The following rotations utils are developed with the help of Gemini.


def vec_quat2euler(quat: np.ndarray) -> np.ndarray:
    """Calculate euler angles form quaternions. This function is vectorized.

    :param quat: quaternions (representation: (w,x,y,z)). Shape: (num_samples,4)
    :return: euler angles (xyz, intrinsic). Shape: (num_samples,3)
    """
    if quat.shape == (4,):
        quat = quat.reshape((1, 4))

    assert quat.shape[1] == 4

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    # calculate roll
    sin_phi = 2.0 * (w * x - y * z)
    cos_phi = w * w - x * x - y * y + z * z
    roll = np.arctan2(sin_phi, cos_phi)

    # calculate pitch
    sin_theta = 2.0 * (w * y + x * z)
    pitch = np.arcsin(np.clip(sin_theta, -1.0, 1.0))

    # calculate yaw
    sin_psi = 2.0 * (w * z - x * y)
    cos_psi = w * w + x * x - y * y - z * z
    yaw = np.arctan2(sin_psi, cos_psi)

    euler = np.stack([roll, pitch, yaw], axis=1)
    return euler


def vec_euler2quat(euler: np.ndarray) -> np.ndarray:
    """Calculate quaternions from euler angles. This function is vectorized.

    :param euler: euler angles (xyz, intrinsic). Shape: (num_samples,3)
    :return: normalized quaternions (representation: (w,x,y,z)). Shape: (num_samples,4)
    """
    ai, aj, ak = euler[:, 0], euler[:, 1], euler[:, 2]
    hi, hj, hk = ai / 2.0, aj / 2.0, ak / 2.0

    ci, si = np.cos(hi), np.sin(hi)
    cj, sj = np.cos(hj), np.sin(hj)
    ck, sk = np.cos(hk), np.sin(hk)

    # quaternion composition for XYZ representation
    # q = q_x * q_y * q_z
    q = np.empty((euler.shape[0], 4))
    q[:, 0] = ci * cj * ck - si * sj * sk  # w
    q[:, 1] = si * cj * ck + ci * sj * sk  # x
    q[:, 2] = ci * sj * ck - si * cj * sk  # y
    q[:, 3] = ci * cj * sk + si * sj * ck  # z

    # ensure that all quaternions are normalized
    q = np.divide(q, np.linalg.norm(q, axis=1, keepdims=True))

    return q


def vec_mat2quat(mats: np.ndarray) -> np.ndarray:
    """Calculate quaternions from rotation matrices. This function is vectorized.

    :param mats: rotation matrices. Shape: (num_samples,3,3)
    :return: the quaternions (representation: (w,x,y,z)) calculated from the rotation matrices. Shape: (num_samples,4)
    """
    R = np.atleast_3d(mats)
    N = R.shape[0]

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # use four potential solutions based on which diagonal element is largest to prevent division by near-zero values.
    q = np.empty((N, 4))

    # trace is positive
    mask0 = tr > 0
    if np.any(mask0):
        S = np.sqrt(tr[mask0] + 1.0) * 2
        q[mask0, 0] = 0.25 * S
        q[mask0, 1] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / S
        q[mask0, 2] = (R[mask0, 0, 2] - R[mask0, 2, 0]) / S
        q[mask0, 3] = (R[mask0, 1, 0] - R[mask0, 0, 1]) / S

    # R[0,0] is the largest diagonal element
    mask1 = (~mask0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if np.any(mask1):
        S = np.sqrt(1.0 + R[mask1, 0, 0] - R[mask1, 1, 1] - R[mask1, 2, 2]) * 2
        q[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S
        q[mask1, 1] = 0.25 * S
        q[mask1, 2] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / S
        q[mask1, 3] = (R[mask1, 0, 2] + R[mask1, 2, 0]) / S

    # R[1,1] is the largest diagonal element
    mask2 = (~mask0) & (~mask1) & (R[:, 1, 1] > R[:, 2, 2])
    if np.any(mask2):
        S = np.sqrt(1.0 + R[mask2, 1, 1] - R[mask2, 0, 0] - R[mask2, 2, 2]) * 2
        q[mask2, 0] = (R[mask2, 0, 2] - R[mask2, 2, 0]) / S
        q[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S
        q[mask2, 2] = 0.25 * S
        q[mask2, 3] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / S

    # R[2,2] is the largest
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if np.any(mask3):
        S = np.sqrt(1.0 + R[mask3, 2, 2] - R[mask3, 0, 0] - R[mask3, 1, 1]) * 2
        q[mask3, 0] = (R[mask3, 1, 0] - R[mask3, 0, 1]) / S
        q[mask3, 1] = (R[mask3, 0, 2] + R[mask3, 2, 0]) / S
        q[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S
        q[mask3, 3] = 0.25 * S

    return q
