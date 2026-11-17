import mujoco
import numpy as np
import pytest

from magbotsim.utils.impedance_control import MoverImpedanceController

MOVER_HALF_HEIGHT = 0.006
MOVER_START_POS = np.array([0.24, 0.24, 0.5])
MOVER_MASS = 1.24
# inertia tensor of the mover in the mover frame (ixx, iyy, izz, ixy, ixz, iyz); the off-diagonal entries ensure that the principal
# axes frame of the mover (body_iquat) is not aligned with the mover frame
MOVER_FULL_INERTIA = np.array([2.5e-3, 3.0e-3, 5.0e-3, 2.0e-4, 1.0e-4, -1.5e-4])
MOVER_INERTIA_DIAG = MOVER_FULL_INERTIA[:3]


def _build_model(actuator_xml_str: str = '') -> mujoco.MjModel:
    full_inertia = ' '.join(str(v) for v in MOVER_FULL_INERTIA)
    start_pos = ' '.join(str(v) for v in MOVER_START_POS)
    xml_str = f"""
    <mujoco>
        <option gravity="0 0 0" timestep="0.002"/>
        <worldbody>
            <body name="mover_0" pos="{start_pos}">
                <joint name="mover_joint_0" type="free"/>
                <inertial pos="0 0 0" mass="{MOVER_MASS}" fullinertia="{full_inertia}"/>
                <geom type="box" size="0.0775 0.0775 {MOVER_HALF_HEIGHT}" contype="0" conaffinity="0"/>
            </body>
        </worldbody>
        <actuator>{actuator_xml_str}
        </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml_str)


def _setup(**controller_kwargs) -> tuple[mujoco.MjModel, mujoco.MjData, MoverImpedanceController]:
    """Create the controller, add its actuators to the model and update the cached MuJoCo data (same procedure as in the envs)."""
    model = _build_model()
    controller = MoverImpedanceController(
        model=model, mover_joint_name='mover_joint_0', mover_half_height=MOVER_HALF_HEIGHT, **controller_kwargs
    )
    model = _build_model(controller.generate_actuator_xml_string(idx_mover=0))
    controller.update_cached_mujoco_data(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, controller


def _expected_damping(translational_stiffness: np.ndarray, rotational_stiffness: np.ndarray, additional_mass: float = 0.0):
    trans = 2 * np.sqrt(translational_stiffness * (MOVER_MASS + additional_mass))
    rot = 2 * np.sqrt(rotational_stiffness * MOVER_INERTIA_DIAG)
    return np.diag(np.concatenate((trans, rot)))


def _rotate_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    return quat


def _simulate_rotation_step(rotational_stiffness: float, axis_idx: int, init_angle: float, duration: float) -> np.ndarray:
    """Start the mover with an orientation error about one axis of the mover frame and control it back to the identity orientation.

    :return: the orientation error (rotation vector in the mover frame) at each time step, i.e. a numpy array of shape (num_steps, 3)
    """
    model, data, controller = _setup(translational_stiffness=1000.0, rotational_stiffness=rotational_stiffness)
    axis = np.zeros(3)
    axis[axis_idx] = 1.0
    data.qpos[3:7] = _rotate_quat(axis, init_angle)
    mujoco.mj_forward(model, data)

    pos_d = MOVER_START_POS - np.array([0.0, 0.0, MOVER_HALF_HEIGHT])
    quat_d = np.array([1.0, 0.0, 0.0, 0.0])
    errors = []
    for _ in range(int(duration / model.opt.timestep)):
        controller.update(model, data, pos_d=pos_d, quat_d=quat_d)
        mujoco.mj_step(model, data)
        err = np.zeros(3)
        mujoco.mju_subQuat(err, data.qpos[3:7], quat_d)
        errors.append(err)
    return np.array(errors)


def test_mover_inertia_in_mover_frame():
    _, _, controller = _setup()
    assert np.allclose(controller.mover_inertia, MOVER_INERTIA_DIAG)


@pytest.mark.parametrize(
    'translational_stiffness, rotational_stiffness',
    [
        (1.0, 0.1),  # default values
        (np.array([5000.0, 5000.0, 10.0]), np.array([5.0, 5.0, 0.0])),
        (np.array([100.0, 200.0, 300.0]), np.array([1.0, 2.0, 3.0])),
    ],
)
def test_damping_matrix_critically_damped(translational_stiffness, rotational_stiffness):
    """Translational damping is based on the mover mass, rotational damping on the moments of inertia of the mover."""
    _, _, controller = _setup(translational_stiffness=translational_stiffness, rotational_stiffness=rotational_stiffness)
    expected = _expected_damping(np.broadcast_to(translational_stiffness, (3,)), np.broadcast_to(rotational_stiffness, (3,)))
    assert np.allclose(controller.damping, expected)


@pytest.mark.parametrize('additional_mass', [0.0, 0.5, 2.0])
def test_additional_mass_only_affects_translational_damping(additional_mass):
    translational_stiffness = np.array([100.0, 200.0, 300.0])
    rotational_stiffness = np.array([1.0, 2.0, 3.0])
    model, data, controller = _setup(translational_stiffness=translational_stiffness, rotational_stiffness=rotational_stiffness)

    controller.update(model, data, pos_d=MOVER_START_POS.copy(), quat_d=np.array([1.0, 0.0, 0.0, 0.0]), additional_mass=additional_mass)
    expected = _expected_damping(translational_stiffness, rotational_stiffness, additional_mass=additional_mass)
    assert np.allclose(controller.damping, expected)


@pytest.mark.parametrize('axis_idx', [0, 1, 2])
def test_rotation_converges_without_overshoot(axis_idx):
    """With critical damping, a mover with an initial orientation error converges back to the desired orientation within a few
    natural periods and without (significant) overshoot. With a damping that is computed from the mass, the rotational DoFs are
    heavily overdamped and the orientation error barely decreases in the same time.
    """
    init_angle = 0.1
    errors = _simulate_rotation_step(rotational_stiffness=0.1, axis_idx=axis_idx, init_angle=init_angle, duration=1.5)
    assert np.all(np.isfinite(errors))
    # error about the excited axis decreases to less than 2% of the initial error
    assert abs(errors[-1, axis_idx]) < 0.02 * init_angle
    # no significant overshoot
    assert np.min(errors[:, axis_idx]) > -0.02 * init_angle


@pytest.mark.parametrize('axis_idx', [0, 1])
def test_rotation_stable_for_high_rotational_stiffness(axis_idx):
    """For a high rotational stiffness, a damping that is computed from the mass is several orders of magnitude too large for the small
    rotational inertia of the mover, which makes the controller numerically unstable. Critical damping based on the inertia keeps it
    stable.
    """
    init_angle = 0.1
    errors = _simulate_rotation_step(rotational_stiffness=5.0, axis_idx=axis_idx, init_angle=init_angle, duration=0.5)
    assert np.all(np.isfinite(errors))
    assert np.max(np.abs(errors)) <= init_angle + 1e-6
    assert np.max(np.abs(errors[-1])) < 0.02 * init_angle
