##########################################################
# Copyright (c) 2024 Lara Bergmann, Bielefeld University #
##########################################################

import numpy as np
import mujoco
from mujoco import MjModel, MjData
from magbotsim.utils import mujoco_utils
from scipy.spatial.transform import Rotation as R


class MoverImpedanceController:
    """A mover impedance controller which solves a position and orientation task for one mover.

    :param model: mjModel of the MuJoCo environment
    :param mover_joint_name: the name of the joint of the mover in the MuJoCo model
    :param joint_mask: None or a numpy array of shape (6,) which contains only 0 and 1, defaults to None.
        This array can be used to control only certain DoFs of a mover with this controller (1 for controlling a DoF).
        If set to None, all DoFs are controlled.
    :param translational_stiffness: a numpy array of shape (3,) or a single float value, defaults to 1.0.
        Use the numpy array to specify different stiffness values for x,y and z. If only a single float value is specified, the
        same stiffness value is used for all translational axes.
    :param rotational_stiffness: a numpy array of shape (3,) or a single float value, defaults to 0.1.
        Use the numpy array to specify different stiffness values for a (rotation about x-axis of the mover frame),
        b (rotation about y-axis of the mover frame) and c (rotation about z-axis of the mover frame). If only a single float value
        is specified, the same stiffness value is used for all rotational axes.
    """

    def __init__(
        self,
        model: MjModel,
        mover_joint_name: str,
        joint_mask: np.ndarray | None = None,
        translational_stiffness: np.ndarray | float = 1.0,
        rotational_stiffness: np.ndarray | float = 0.1,
    ) -> None:
        self.mover_joint_name = mover_joint_name
        self.mover_body_id = model.joint(self.mover_joint_name).bodyid[0]
        self.mover_dofadr = model.body(self.mover_body_id).dofadr[0]
        self.mover_dofnum = model.body(self.mover_body_id).dofnum[0]

        # configure stiffness matrix
        self.stiffness = np.zeros((6, 6))
        self.stiffness[:3, :3] = self.init_stiffness_mat(stiffness=translational_stiffness)
        self.stiffness[-3:, -3:] = self.init_stiffness_mat(stiffness=rotational_stiffness)
        # configure damping matrix (damping ratio = 1)
        mover_mass = model.body(self.mover_body_id).mass[0]
        self.damping = 2 * np.sqrt(self.stiffness * mover_mass)

        # enable/disable joints
        if joint_mask is None:
            self.joint_mask = np.ones(6).astype(np.bool)
        else:
            assert joint_mask.shape == (6,)
            assert (np.bitwise_or(joint_mask == 0, joint_mask == 1)).all()
            self.joint_mask = joint_mask.astype(np.bool)

    def init_stiffness_mat(self, stiffness: np.ndarray | float) -> np.ndarray:
        """Initialize a stiffness matrix.

        :param stiffness: the stiffness values - either a numpy array of shape (3,) or a single float value
        :return: a numpy array of shape (3,3) which is a diagonal matrix with the stiffness values on its diagonal
        """
        if isinstance(stiffness, float):
            return np.eye(3, 3) * stiffness
        else:
            assert isinstance(stiffness, np.ndarray)
            assert stiffness.shape == (3,)
            return np.diag(stiffness)

    def set_joint_mask(self, new_joint_mask: np.ndarray):
        """Set a new joint mask to control only specified DoFs of a mover with this controller.

        :param new_joint_mask: the new joint mask - a numpy array of shape (6,) which contains only 0 and 1 (1 for controlling a DoF).
        """
        assert new_joint_mask.shape == (6,)
        assert (np.bitwise_or(new_joint_mask == 0, new_joint_mask == 1)).all()
        self.joint_mask = new_joint_mask.astype(np.bool)

    def generate_actuator_xml_string(self, idx_mover: int):
        """Generate an actuator XML string which can be added to the MuJoCo model XML string. Note that only actuators for DoFs that
        are controlled by this controller are added. This method must be called manually by the user after the controller has been
        initialized.

        :param idx_mover: the index of the mover (can be found in the body name of the mover, e.g. for a mover with index 0: 'mover_0')
        :return: the actuator XML string
        """
        names = ['x', 'y', 'z', 'a', 'b', 'c']
        actuator_xml_lines = []
        self.actuator_names = []

        for idx, (name, mask) in enumerate(zip(names, self.joint_mask, strict=True)):
            if mask:
                str_gear = ' '.join('1' if i == idx else '0' for i in range(6))
                actuator_name = f'mover_actuator_{name}_{idx_mover}'
                self.actuator_names.append(actuator_name)
                actuator_xml_lines.append(
                    f'\n\t\t<general name="{actuator_name}" joint="{self.mover_joint_name}" '
                    f'gear="{str_gear}" dyntype="none" gaintype="fixed" biastype="none"/>'
                )
            else:
                self.actuator_names.append('')

        return ''.join(actuator_xml_lines)

    def update_cached_actuator_mujoco_data(self, model: MjModel) -> None:
        """Update all cached information about MuJoCo actuators, such as names and ids.

        :param model: mjModel of the MuJoCo environment
        """
        self.actuator_ids = np.zeros((len(self.actuator_names),), dtype=np.int32)
        for idx_a, actuator_name in enumerate(self.actuator_names):
            if len(actuator_name) > 0:
                self.actuator_ids[idx_a] = model.actuator(actuator_name).id

    def ctrl_callback(self, ctrl: np.ndarray):
        """A callback that can be used to modify the desired forces and torques computed by the ``update()`` method, e.g. to ensure
        minimum and maximum forces and torques.

        :param ctrl: the desired controls (forces, torques) computed by the 'update()' method, i.e. a numpy array of shape (6,) with
            the following order of DoFs: x,y,z,a,b,c.
        :return: the modified controls, i.e. a numpy array of shape (6,) with the following order of DoFs: x,y,z,a,b,c.
        """
        return ctrl

    def update(self, model: MjModel, data: MjData, pos_d: np.ndarray, quat_d: np.ndarray) -> None:
        """Compute new controls based on the position and orientation error.

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        :param pos_d: the desired position (x_p,y_p,z_p) specified as a numpy array of shape (3,)
        :param quat_d: the desired orientation, specified as a quaternion (w_o,x_o,y_o,z_o), i.e. numpy array of shape (4,)
        """
        assert pos_d.shape == (3,)
        assert quat_d.shape == (4,)

        # desired rot mat
        r_quat = R.from_quat(quat_d, scalar_first=True)
        xmat_d = r_quat.as_matrix()

        # joint velocities
        dq = mujoco_utils.get_joint_qvel(model, data, self.mover_joint_name).reshape((-1, 1))

        # Jacobians
        jacp = np.zeros((3, model.nv))  # tanslational part of the Jacobian
        jacr = np.zeros((3, model.nv))  # rotational part of the Jacobian
        mujoco.mj_jacBody(model, data, jacp, jacr, self.mover_body_id)
        jac = np.vstack((jacp, jacr))
        jac = jac[:, self.mover_dofadr : self.mover_dofadr + self.mover_dofnum]

        # get Cartesian position and orientation of the mover
        xpos = data.xpos[self.mover_body_id, :]
        xmat = data.xmat[self.mover_body_id, :].reshape(3, 3)

        error = np.zeros((6, 1))
        # position error
        error[:3, 0] = pos_d - xpos
        # orientation error
        r_mat = R.from_matrix(xmat.T @ xmat_d)
        rot_vec = r_mat.as_rotvec()
        error[-3:, :] = xmat @ rot_vec.reshape((3, 1))  # ee frame orientation -> base frame orientation

        # compute controls
        ctrl = self.joint_mask.astype(np.int64) * (jac.T @ (self.stiffness @ error - self.damping @ (jac @ dq))).flatten()

        # modify the computed controls, if desired
        ctrl = self.ctrl_callback(ctrl=ctrl.copy())

        # set controls
        if hasattr(self, 'actuator_ids'):
            data.ctrl[self.actuator_ids[self.joint_mask]] = ctrl[self.joint_mask]
        else:
            for idx in range(0, 6):
                if self.joint_mask[idx]:
                    mujoco_utils.set_actuator_ctrl(model, data, actuator_name=self.actuator_names[idx], value=ctrl[idx])
