from dataclasses import dataclass
import numpy as np
import mujoco
from mujoco import MjModel, MjData
from scipy.interpolate import CubicSpline
from magbotsim.utils import rotations_utils, mujoco_utils
from magbotsim.utils.rotations_utils import ContinuousYawExtractor


@dataclass(frozen=True)
class GripperMagBotsAPM4330Params:
    n: float = 0.193

    z_b01: float = 0.05811
    z_t01: float = 0.0484
    x_b01: float = 0.03
    x_t01: float = 0.066
    k_01: float = 0.182

    z_b2: float = 0.06801
    z_t2: float = 0.0385
    x_b2: float = 0.032
    x_t2: float = 0.064
    k_2: float = 0.11386

    s_m0: float = 0.9969
    s_m1: float = 0.9969
    s_m2: float = -182.4376

    z_p_min: float = 0.167732
    z_p_max: float = 0.22037
    z_d_min: float = 0.026
    f_d_max: float = 0.017


class GripperMagBotsAPM4330:
    """Base class for Gripper MagBots. This class can handle multiple Gripper MagBots, allowing calculations to be vectorized.

    :param num_magbots: number of Gripper MagBots
    :param indices_mover_c0: a numpy array of shape (num_magbots,) containing all indices of movers attached to connection 0
    :param indices_mover_c1: a numpy array of shape (num_magbots,) containing all indices of movers attached to connection 1
    :param indices_mover_c2: a numpy array of shape (num_magbots,) containing all indices of movers attached to connection 2
    :param magbot_params: an instance of GripperMagBotsAPM4330Params
    :param cycle_time: the cycle time used in the BasicMagBotEnv (usually 1ms)
    """

    def __init__(
        self,
        num_magbots: int,
        indices_mover_c0: np.ndarray,
        indices_mover_c1: np.ndarray,
        indices_mover_c2: np.ndarray,
        magbot_params: GripperMagBotsAPM4330Params,
        cycle_time: float,
    ) -> None:
        self.num_magbots = num_magbots
        self.magbots = []
        self.magbot_params = magbot_params

        # mover indices
        self.mover_indices = np.concatenate((indices_mover_c0[:, None], indices_mover_c1[:, None], indices_mover_c2[:, None]), axis=1)
        self.indices_mover_c0 = indices_mover_c0
        self.indices_mover_c1 = indices_mover_c1
        self.indices_mover_c2 = indices_mover_c2

        # gripper qpos
        self.gripper_a_joint_names = [f'gripper_alpha_joint_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]
        self.gripper_b_joint_names = [f'gripper_beta_joint_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]
        self.gripper_f_joint_names = [f'gripper_jaw_1_hinge_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]

        self.gripper_pose_site_names = [f'gripper_pose_site_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]

        # gripper finger pads (used to measure the distance between the two fingers)
        self.gripper_pad_1_geom_names = [f'gripper_pad_1_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]
        self.gripper_pad_2_geom_names = [f'gripper_pad_2_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]
        # calibration mapping between the finger distance and the finger (jaw) angle, built in update_cached_mujoco_data
        self._finger_distance_grid = None
        self._finger_angle_grid = None
        self._finger_distance_to_angle_spline: CubicSpline | None = None
        self.finger_distance_min = None
        self.finger_distance_max = None

        # couple mover rotation to gripper rotations
        self.cycle_time = cycle_time
        self.mover_yaw_estimator = None
        self.Kp = 3.0
        self.Ki = 0.1
        self.Kd = 0.6

    def _update_cached_gripper_site_ids(self, model: MjModel) -> None:
        """Update gripper site ids.

        :param model: mjModel of the MuJoCo environment
        """
        self.gripper_pose_site_ids = np.zeros((self.num_magbots,), dtype=np.int32)
        for idx_magbot in range(0, self.num_magbots):
            self.gripper_pose_site_ids[idx_magbot] = model.site(self.gripper_pose_site_names[idx_magbot]).id

    def _update_cached_gripper_finger_geom_ids(self, model: MjModel) -> None:
        """Update the geom ids of the two finger pads (used to measure the distance between the fingers).

        :param model: mjModel of the MuJoCo environment
        """
        self.gripper_pad_1_geom_ids = np.zeros((self.num_magbots,), dtype=np.int32)
        self.gripper_pad_2_geom_ids = np.zeros((self.num_magbots,), dtype=np.int32)
        for idx_magbot in range(0, self.num_magbots):
            self.gripper_pad_1_geom_ids[idx_magbot] = model.geom(self.gripper_pad_1_geom_names[idx_magbot]).id
            self.gripper_pad_2_geom_ids[idx_magbot] = model.geom(self.gripper_pad_2_geom_names[idx_magbot]).id

    def _build_finger_distance_calibration(self, model: MjModel, num_samples: int = 25, settle_steps: int = 400) -> None:
        """Build the calibration mapping between the distance of the two finger pads and the finger (jaw) angle.

        :param model: mjModel of the MuJoCo environment
        :param num_samples: number of samples used to build the calibration, defaults to 25
        :param settle_steps: number of simulation steps used to let the finger kinematics settle per sample, defaults to 400
        """
        finger_actuator_id = model.actuator('gripper_actuator_f_0').id
        pad_1_geom_id = self.gripper_pad_1_geom_ids[0]
        pad_2_geom_id = self.gripper_pad_2_geom_ids[0]
        ctrl_lo, ctrl_hi = model.actuator_ctrlrange[finger_actuator_id]
        finger_angles = np.linspace(ctrl_lo, ctrl_hi, num_samples)
        finger_distances = np.zeros((num_samples,))

        # settle the finger kinematics on a scratch copy of the model with gravity disabled, so that the static finger
        # opening only depends on the commanded finger angle and not on the (uncontrolled) pose of the rest of the MagBot
        calib_data = MjData(model)
        gravity = model.opt.gravity.copy()
        model.opt.gravity[:] = 0.0
        try:
            for idx_sample, finger_angle in enumerate(finger_angles):
                mujoco.mj_resetData(model, calib_data)
                calib_data.ctrl[finger_actuator_id] = finger_angle
                for _ in range(0, settle_steps):
                    mujoco.mj_step(model, calib_data)
                finger_distances[idx_sample] = mujoco.mj_geomDistance(model, calib_data, pad_1_geom_id, pad_2_geom_id, 1.0, None)
        finally:
            model.opt.gravity[:] = gravity

        # store the samples ordered by ascending distance (required by CubicSpline)
        order = np.argsort(finger_distances)
        self._finger_distance_grid = finger_distances[order]
        self._finger_angle_grid = finger_angles[order]
        self._finger_distance_to_angle_spline = CubicSpline(self._finger_distance_grid, self._finger_angle_grid)
        self.finger_distance_min = float(self._finger_distance_grid[0])
        self.finger_distance_max = float(self._finger_distance_grid[-1])

    def finger_distance_to_angle(self, finger_distance: np.ndarray) -> np.ndarray:
        """Convert a desired distance between the two finger pads to the corresponding finger (jaw) angle.

        The desired distance is clipped to the reachable finger distance range (``finger_distance_min``, ``finger_distance_max``).

        :param finger_distance: the desired distance(s) between the finger pads in m. Shape: (num_samples,)
        :return: the corresponding finger (jaw) angle(s) in rad. Shape: (num_samples,)
        """
        if self._finger_distance_grid is None or self._finger_distance_to_angle_spline is None:
            raise RuntimeError(
                'Finger distance calibration is not available. Call update_cached_mujoco_data(model) before using '
                'finger_distance_to_angle or gripperSetPose2MoverSetPose.'
            )

        clipped_distance = np.clip(finger_distance, self._finger_distance_grid[0], self._finger_distance_grid[-1])
        return self._finger_distance_to_angle_spline(clipped_distance)

    def _update_cached_gripper_qpos_indices(self, model: MjModel) -> None:
        """Update gripper qpos indices.

        :param model: mjModel of the MuJoCo environment
        """
        # A
        self.gripper_qpos_a_indices = np.zeros((len(self.gripper_a_joint_names),), dtype=np.int32)
        for idx_joint, joint_name in enumerate(self.gripper_a_joint_names):
            joint_qpos_adr, _, qpos_ndim, qvel_qacc_ndim = mujoco_utils.get_joint_addrs_and_ndims(model, joint_name)
            assert qpos_ndim == 1 and qvel_qacc_ndim == 1, 'Gripper a joint ndims are not as expected'
            self.gripper_qpos_a_indices[idx_joint] = joint_qpos_adr
        # B
        self.gripper_qpos_b_indices = np.zeros((len(self.gripper_b_joint_names),), dtype=np.int32)
        for idx_joint, joint_name in enumerate(self.gripper_b_joint_names):
            joint_qpos_adr, _, qpos_ndim, qvel_qacc_ndim = mujoco_utils.get_joint_addrs_and_ndims(model, joint_name)
            assert qpos_ndim == 1 and qvel_qacc_ndim == 1, 'Gripper b joint ndims are not as expected'
            self.gripper_qpos_b_indices[idx_joint] = joint_qpos_adr
        # F
        self.gripper_qpos_f_indices = np.zeros((len(self.gripper_f_joint_names),), dtype=np.int32)
        for idx_joint, joint_name in enumerate(self.gripper_f_joint_names):
            joint_qpos_adr, _, qpos_ndim, qvel_qacc_ndim = mujoco_utils.get_joint_addrs_and_ndims(model, joint_name)
            assert qpos_ndim == 1 and qvel_qacc_ndim == 1, 'Gripper f joint ndims are not as expected'
            self.gripper_qpos_f_indices[idx_joint] = joint_qpos_adr

    def _update_cached_actuator_ids(self, model: MjModel) -> None:
        """Update actuator ids.

        :param model: mjModel of the MuJoCo environment
        """
        # mover actuators
        self.mover_actuator_ids = np.zeros((self.num_magbots * 3,), dtype=np.int32)
        for idx_mover in range(0, self.num_magbots * 3):
            self.mover_actuator_ids[idx_mover] = model.actuator(self.mover_actuator_names[idx_mover]).id

        # gripper actuators
        self.gripper_abf_actuator_ids = np.zeros((self.num_magbots, 3), dtype=np.int32)
        for idx_magbot in range(0, self.num_magbots):
            for idx_a, actuator_name in enumerate(self.gripper_abf_actuator_names[idx_magbot]):
                self.gripper_abf_actuator_ids[idx_magbot, idx_a] = model.actuator(actuator_name).id

    def reset_yaw_extractor(self) -> None:
        """Reset the continuous mover yaw extractor, i.e. set all integrated valued to zero."""
        self.mover_yaw_extractor = ContinuousYawExtractor(num_samples=self.num_magbots * 3)
        # the gripper c-rotation is derived from an arctan2 and therefore has to be unwrapped, otherwise the gripper
        # b-rotation and finger angle, which are both derived from c, jump by 2*pi whenever c crosses +/-pi
        self.gripper_c_unwrapper = ContinuousYawExtractor(num_samples=self.num_magbots)
        self.error_integrated = np.zeros((self.num_magbots * 3,))

    def update_cached_mujoco_data(self, model: MjModel) -> None:
        """Update all cached information about MuJoCo objects, such as names and IDs.

        :param model: mjModel of the MuJoCo environment
        """
        # actuator IDs
        self._update_cached_actuator_ids(model)
        # gripper
        self._update_cached_gripper_qpos_indices(model)
        # gripper site IDs
        self._update_cached_gripper_site_ids(model)
        # gripper finger pad geom IDs
        self._update_cached_gripper_finger_geom_ids(model)
        # finger distance to finger angle calibration
        self._build_finger_distance_calibration(model)
        # MagBot mass
        self.magbot_masses = np.zeros((self.num_magbots,))
        for idx_magbot in range(0, self.num_magbots):
            body_name = f'gripper_magbot_{idx_magbot}'
            self.magbot_masses[idx_magbot] = model.body(body_name).subtreemass[0]
        # update yaw estimator
        self.reset_yaw_extractor()

    def control_gripper(
        self, model: MjModel, data: MjData, mover_poses_d: np.ndarray, current_mover_qpos: np.ndarray, current_mover_qvels: np.ndarray
    ) -> None:
        """Couple the rotations around the z-axes of the movers with the grippers's a and b rotations, and the opening of the fingers.
        This method is vectorized and can therefore couple the rotations for multiple MagBots.

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        :param mover_poses_d: the desired mover poses (rotation specified as Euler angles (xyz) in rad). Shape: (num_magbots*3,6)
        :param current_mover_qpos: the current qpos of all movers (rotation specified as quaternion [w,x,y,z]). Shape: (num_magbots*3,7)
        :param current_mover_qvels: the current qvel of all movers. Shape: (num_magbots*3,6)
        """
        if not hasattr(self, 'gripper_abf_actuator_ids'):
            self._update_cached_actuator_ids(model)
        if len(mover_poses_d.shape) > 2:
            mover_poses_d = mover_poses_d.reshape((-1, 6))

        quats = current_mover_qpos[:, 3:7]
        mover_yaws = self.mover_yaw_extractor.update(quats)

        xyc_c0_movers = np.zeros((self.num_magbots, 3))
        xyc_c1_movers = np.zeros((self.num_magbots, 3))
        xyc_c2_movers = np.zeros((self.num_magbots, 3))

        xyc_c0_movers[:, :2] = current_mover_qpos[self.indices_mover_c0, :2]
        xyc_c1_movers[:, :2] = current_mover_qpos[self.indices_mover_c1, :2]
        xyc_c2_movers[:, :2] = current_mover_qpos[self.indices_mover_c2, :2]

        xyc_c0_movers[:, 2] = mover_yaws[self.indices_mover_c0]
        xyc_c1_movers[:, 2] = mover_yaws[self.indices_mover_c1]
        xyc_c2_movers[:, 2] = mover_yaws[self.indices_mover_c2]

        # platform xyz-pos
        x_p = np.divide((xyc_c0_movers[:, 0] + xyc_c1_movers[:, 0]), 2)
        y_p = np.divide((xyc_c0_movers[:, 1] + xyc_c1_movers[:, 1]), 2)

        # gripper rotations
        a = -0.5 * (np.divide(xyc_c0_movers[:, 2], self.magbot_params.s_m0) - np.divide(xyc_c1_movers[:, 2], self.magbot_params.s_m1))
        c = self.gripper_c_unwrapper.update_from_angle(np.arctan2(xyc_c1_movers[:, 1] - y_p, xyc_c1_movers[:, 0] - x_p))
        b = a - np.divide(xyc_c1_movers[:, 2] - c, self.magbot_params.s_m1)

        # gripper finger
        f = np.divide(xyc_c2_movers[:, 2] - a - c, self.magbot_params.s_m2)

        data.ctrl[self.gripper_abf_actuator_ids[:, 0]] = a
        data.ctrl[self.gripper_abf_actuator_ids[:, 1]] = b
        data.ctrl[self.gripper_abf_actuator_ids[:, 2]] = f

        # set mover c pose
        yaw_error = mover_poses_d[:, -1] - mover_yaws
        self.error_integrated += yaw_error * self.cycle_time
        integral_term = np.clip(self.Ki * self.error_integrated, -1.0, 1.0)
        data.ctrl[self.mover_actuator_ids] = self.Kp * yaw_error + integral_term - self.Kd * current_mover_qvels[:, 5]

    def generate_magbot_xml_strings(self, initial_pos_xyz_mover_c0: np.ndarray, custom_model_xml_strings: dict | None) -> dict:
        """Generate the MuJoCo XML strings for all Gripper MagBots and add them to the ``custom_model_xml_strings``-dict.

        :param initial_pos_xyz_mover_c0: the initial x,y,z-positions for all movers attached to connection 0. Shape: (num_magbots,3)
        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this method
        :return: the modified ``custom_model_xml_strings``-dict
        """
        assert initial_pos_xyz_mover_c0.shape == (self.num_magbots, 3)

        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}

        for idx_magbot in range(0, self.num_magbots):
            if idx_magbot == 0:
                # option
                custom_option_xml_str = (
                    '\t<option timestep="0.001" cone="elliptic" jacobian="dense" gravity="0 0 -9.81" '
                    'integrator="implicitfast"><flag diagexact="enable"/></option>'
                )
                custom_model_xml_strings['custom_option_xml_str'] = custom_option_xml_str
                # assets
                gripper_assets_list = [
                    '\n\t\t<!-- gripper magbot assets -->',
                    '\t\t<material name="light_gray" specular="1" shininess="1" reflectance="1" rgba="0.8 0.8 0.8 1"/>',
                    '\t\t<material name="mat_tan" specular="1" shininess="1" reflectance="1" rgba="0.941176 0.87451  0.678431 1"/>',
                    '\t\t<material name="mat_amber" specular="0.1" shininess="0.01" reflectance="0.01" '
                    'rgba="0.980392 0.796078 0.262745 1"/>',
                    '\t\t<mesh name="mover_adapter" file="magbots/gripper_XPlanar_APM4330/mover_adapter.stl" scale="1 1 1" />',
                    '\t\t<mesh name="cover_base" file="magbots/gripper_XPlanar_APM4330/cover_base.stl" scale="1 1 1" />',
                    '\t\t<mesh name="spring_loaded_pin" file="magbots/gripper_XPlanar_APM4330/spring_loaded_pin.stl" scale="1 1 1" />',
                    '\t\t<mesh name="rail_mount_m0_mesh" content_type="model/stl" '
                    'file="magbots/gripper_XPlanar_APM4330/rail_mount_m0.stl"/>',
                    '\t\t<mesh name="rail_mount_m1m2_mesh" content_type="model/stl" '
                    'file="magbots/gripper_XPlanar_APM4330/rail_mount_m1m2.stl"/>',
                    '\t\t<mesh name="rotation_plate_m0" file="./magbots/gripper_XPlanar_APM4330/rotation_plate_m1.stl" scale="1 1 1" />',
                    '\t\t<mesh name="rotation_plate_m1" file="./magbots/gripper_XPlanar_APM4330/rotation_plate_m2.stl" scale="1 1 1" />',
                    '\t\t<mesh name="rotation_plate_m2" file="./magbots/gripper_XPlanar_APM4330/rotation_plate_m3.stl" scale="1 1 1" />',
                    '\t\t<mesh name="slide_bearing" file="./magbots/gripper_XPlanar_APM4330/slide_bearing.stl" scale="1 1 1" />',
                    '\t\t<mesh name="shaft_collar_large" file="./magbots/gripper_XPlanar_APM4330/shaft_collar_large.stl" scale="1 1 1" />',
                    '\t\t<mesh name="stainless_steel_shaft_6x103mm" '
                    'file="./magbots/gripper_XPlanar_APM4330/stainless_steel_shaft_6x103mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="stainless_steel_shaft_6x92mm" '
                    'file="./magbots/gripper_XPlanar_APM4330/stainless_steel_shaft_6x92mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="toothed_wheel_1" file="./magbots/gripper_XPlanar_APM4330/toothed_wheel_1.stl" scale="1 1 1" />',
                    '\t\t<mesh name="toothed_wheel_2" file="./magbots/gripper_XPlanar_APM4330/toothed_wheel_2.stl" scale="1 1 1" />',
                    '\t\t<mesh name="HTD3M_pulley" file="./magbots/gripper_XPlanar_APM4330/HTD3M_pulley.stl" scale="1 1 1" />',
                    '\t\t<mesh name="HTD3M_toothed_belt" file="./magbots/gripper_XPlanar_APM4330/HTD3M_toothed_belt.stl" scale="1 1 1" />',
                    '\t\t<mesh name="GT2_pulley_10mm" file="./magbots/gripper_XPlanar_APM4330/GT2_pulley_10mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="GT2_toothed_belt_10mm" file="./magbots/gripper_XPlanar_APM4330/GT2_toothed_belt_10mm.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="GT2_pulley_6mm" file="./magbots/gripper_XPlanar_APM4330/GT2_pulley_6mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="GT2_toothed_belt_6mm" file="./magbots/gripper_XPlanar_APM4330/GT2_toothed_belt_6mm.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="solid_connection_m0m1" file="./magbots/gripper_XPlanar_APM4330/solid_connection_m1m2.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="solid_connection_m2" file="./magbots/gripper_XPlanar_APM4330/solid_connection_m3.stl" scale="1 1 1"/>',
                    '\t\t<mesh name="platform" file="./magbots/gripper_XPlanar_APM4330/platform.stl" scale="1 1 1" />',
                    '\t\t<!-- gripper end-effector assets -->',
                    '\t\t<mesh name="gripper_platform" file="magbots/gripper_XPlanar_APM4330/gripper/gripper_platform.stl" scale="1 1 1"/>',
                    '\t\t<mesh name="gripper_base" file="magbots/gripper_XPlanar_APM4330/gripper/gripper_base.stl" scale="1 1 1" />',
                    '\t\t<mesh name="finger_pad" file="magbots/gripper_XPlanar_APM4330/gripper/finger_pad.stl" scale="1 1 1" />',
                    '\t\t<mesh name="toothed_wheel_large" file="magbots/gripper_XPlanar_APM4330/gripper/toothed_wheel_large.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="toothed_wheel_small" file="magbots/gripper_XPlanar_APM4330/gripper/toothed_wheel_small.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="shaft_3x153mm" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_3x153mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="shaft_collar_wide" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_collar_wide.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="shaft_collar_thin" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_collar_thin.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="shaft_with_hole_long" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_with_hole_long.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="pivot_link" file="magbots/gripper_XPlanar_APM4330/gripper/pivot_link.stl" scale="1 1 1" />',
                    '\t\t<mesh name="finger_coupler" file="magbots/gripper_XPlanar_APM4330/gripper/finger_coupler.stl" scale="1 1 1" />',
                    '\t\t<mesh name="four_bar_link" file="magbots/gripper_XPlanar_APM4330/gripper/four_bar_link.stl" scale="1 1 1" />',
                    '\t\t<mesh name="finger_body" file="magbots/gripper_XPlanar_APM4330/gripper/finger_body.stl" scale="1 1 1" />',
                    '\t\t<mesh name="jaw_link" file="magbots/gripper_XPlanar_APM4330/gripper/jaw_link.stl" scale="1 1 1" />',
                    '\t\t<mesh name="shaft_collar_small" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_collar_small.stl" '
                    'scale="1 1 1" />',
                    '\t\t<!-- gripper mount frame on the platform -->',
                    '\t\t<mesh name="shaft_collar_thick" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_collar_thick.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="shaft_with_hole_short" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_with_hole_short.stl" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="mounting_bracket" file="magbots/gripper_XPlanar_APM4330/gripper/mounting_bracket.stl" scale="1 1 1"/>',
                    '\t\t<mesh name="flanged_shaft" file="magbots/gripper_XPlanar_APM4330/gripper/flanged_shaft.stl" scale="1 1 1" />',
                    '\t\t<mesh name="shaft_6x92mm" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_6x92mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="shaft_6x103mm" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_6x103mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="shaft_3x45mm" file="magbots/gripper_XPlanar_APM4330/gripper/shaft_3x45mm.stl" scale="1 1 1" />',
                    '\t\t<mesh name="flanged_pulley" file="magbots/gripper_XPlanar_APM4330/gripper/flanged_pulley.stl" scale="1 1 1" />',
                ]
                gripper_assets_xml_str = '\n'.join(gripper_assets_list)
                custom_assets_xml_str = custom_model_xml_strings.get('custom_assets_xml_str', None)
                if custom_assets_xml_str is not None:
                    custom_assets_xml_str += gripper_assets_xml_str
                else:
                    custom_assets_xml_str = gripper_assets_xml_str
                custom_model_xml_strings['custom_assets_xml_str'] = custom_assets_xml_str

            # mover body XML strings
            mover_body_xml_strs = custom_model_xml_strings.get('custom_mover_body_xml_str_list')
            assert isinstance(mover_body_xml_strs, list), 'custom_mover_body_xml_str_list must be a list'
            # mover connection 0
            mover_body_c0_list = [
                f'\t\t\t<site name="site_m_{self.indices_mover_c0[idx_magbot]}" pos="0 0 0"/>',
                f'\n\t\t\t<body name="m_{self.indices_mover_c0[idx_magbot]}_rail" pos="0 0 0.006" gravcomp="1" euler="0 0 -1.5708">',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c0[idx_magbot]}_to_mount_geom" type="mesh" mesh="mover_adapter" mass="0.065" '
                'pos="0 0 0" material="black"/>',
                f'\t\t\t\t<site name="m_{self.indices_mover_c0[idx_magbot]}_rail_site" pos="0 0 0" euler="0 0 1.5708" size="0.001" '
                'type="sphere"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c0[idx_magbot]}_pin_1_geom" type="mesh" pos="-0.017 0 0.0125" '
                'euler="0 0 -1.5708" mesh="spring_loaded_pin" material="black"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c0[idx_magbot]}_pin_2_geom" type="mesh" pos="0.017 0 0.0125" '
                'euler="0 0 1.5708" mesh="spring_loaded_pin" material="black"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c0[idx_magbot]}_cover_base" type="mesh" pos="0 0 0.0167" mesh="cover_base" '
                'material="black"/>',
                '\t\t\t</body>',
            ]
            mover_body_c0_xml_str = '\n'.join(mover_body_c0_list)
            mover_body_xml_strs[self.indices_mover_c0[idx_magbot]] = mover_body_c0_xml_str
            # mover connection 1
            mover_body_c1_list = [
                f'\t\t\t<site name="site_m_{self.indices_mover_c1[idx_magbot]}" pos="0 0 0"/>',
                f'\n\t\t\t<body name="m_{self.indices_mover_c1[idx_magbot]}_rail" pos="0 0 0.006" gravcomp="1" euler="0 0 1.5708">',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c1[idx_magbot]}_to_mount_geom" type="mesh" mesh="mover_adapter" mass="0.065" '
                'pos="0 0 0" material="black"/>',
                f'\t\t\t\t<site name="m_{self.indices_mover_c1[idx_magbot]}_rail_site" pos="0 0 0" euler="0 0 1.5708" size="0.001" '
                'type="sphere"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c1[idx_magbot]}_pin_1_geom" type="mesh" pos="-0.017 0 0.0125" '
                'euler="0 0 -1.5708" mesh="spring_loaded_pin" material="black"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c1[idx_magbot]}_pin_2_geom" type="mesh" pos="0.017 0 0.0125" euler="0 0 1.5708"'
                ' mesh="spring_loaded_pin" material="black"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c1[idx_magbot]}_cover_base" type="mesh" pos="0 0 0.0167" mesh="cover_base" '
                'material="black"/>',
                '\t\t\t</body>',
            ]
            mover_body_c1_xml_str = '\n'.join(mover_body_c1_list)
            mover_body_xml_strs[self.indices_mover_c1[idx_magbot]] = mover_body_c1_xml_str
            custom_model_xml_strings['custom_mover_body_xml_str_list'] = mover_body_xml_strs
            # mover connection 2
            mover_body_c2_list = [
                f'\t\t\t<site name="site_m_{self.indices_mover_c2[idx_magbot]}" pos="0 0 0"/>',
                f'\n\t\t\t<body name="m_{self.indices_mover_c2[idx_magbot]}_rail" pos="0 0 0.006" gravcomp="1" euler="0 0 1.5708">',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c2[idx_magbot]}_to_mount_geom" type="mesh" mesh="mover_adapter" mass="0.065" '
                'pos="0 0 0" material="black"/>',
                f'\t\t\t\t<site name="m_{self.indices_mover_c2[idx_magbot]}_rail_site" pos="0 0 0" euler="0 0 -1.5708" size="0.001" '
                'type="sphere"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c2[idx_magbot]}_pin_1_geom" type="mesh" pos="-0.017 0 0.0125" '
                'euler="0 0 -1.5708" mesh="spring_loaded_pin" material="black"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c2[idx_magbot]}_pin_2_geom" type="mesh" pos="0.017 0 0.0125" euler="0 0 1.5708"'
                ' mesh="spring_loaded_pin" material="black"/>',
                f'\t\t\t\t<geom name="m_{self.indices_mover_c2[idx_magbot]}_cover_base" type="mesh" pos="0 0 0.0167" mesh="cover_base" '
                'material="black"/>',
                '\t\t\t</body>',
            ]
            mover_body_c2_xml_str = '\n'.join(mover_body_c2_list)
            mover_body_xml_strs[self.indices_mover_c2[idx_magbot]] = mover_body_c2_xml_str
            custom_model_xml_strings['custom_mover_body_xml_str_list'] = mover_body_xml_strs

            # gripper magbot
            gripper_magbot_list = [
                f'\n\t\t<!-- Gripper MagBot {idx_magbot} -->',
                f'\t\t<body name="gripper_magbot_{idx_magbot}" pos="{initial_pos_xyz_mover_c0[idx_magbot, 0]} '
                f'{initial_pos_xyz_mover_c0[idx_magbot, 1]} {initial_pos_xyz_mover_c0[idx_magbot, 2] + 0.012}" gravcomp="1">',
                '\t\t\t<inertial mass="0.001" diaginertia="1e-6 1e-6 1e-6" pos="0 0 0"/>',
                f'\t\t\t<joint name="gripper_x_{idx_magbot}" type="slide" axis="1 0 0"/>',
                f'\t\t\t<joint name="gripper_y_{idx_magbot}" type="slide" axis="0 1 0"/>',
                f'\t\t\t<joint name="gripper_yaw_{idx_magbot}" type="hinge" axis="0 0 1"/>',
                '\t\t\t<!-- connection 0 -->',
                f'\t\t\t<body name="rail_mount_0_{idx_magbot}" pos="0 0 0" gravcomp="1">',
                f'\t\t\t\t<joint name="rail_mount_0_joint_{idx_magbot}" type="slide" axis="1 0 0" damping="0.1"/>',
                f'\t\t\t\t<joint name="rail_mount_0_joint_yaw_{idx_magbot}" type="hinge" axis="0 0 1"/>',
                f'\t\t\t\t<geom name="rail_mount_geom_0_{idx_magbot}" type="mesh" mesh="rail_mount_m0_mesh" mass="0.5" pos="0 0 0" '
                'material="black"/>',
                f'\t\t\t\t<site name="rail_mount_0_site_{idx_magbot}" pos="0 0 0" size="0.001" type="sphere"/>',
                f'\t\t\t\t<body name="rotation_plate_m0_{idx_magbot}" pos="0 0 0.0285" gravcomp="1">',
                f'\t\t\t\t\t<joint name="rotation_plate_m0_joint_{idx_magbot}" type="hinge" axis="0 0 1" pos="0 0 0" damping="0"/>',
                f'\t\t\t\t\t<geom name="rotation_plate_m0_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" mesh="rotation_plate_m0" '
                'material="black"/>',
                f'\t\t\t\t\t<geom name="slide_bearing_0_m0_{idx_magbot}" type="mesh" pos="0 0.036 0.0334" euler="1.5707963 0 0" '
                'mesh="slide_bearing" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="slide_bearing_1_m0_{idx_magbot}" type="mesh" pos="0 -0.047 0.0334" euler="-1.5707963 0 0" '
                'mesh="slide_bearing" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="stainless_steel_shaft_m0_b_{idx_magbot}" type="mesh" pos="0 -0.0055 0.0334" '
                'mesh="stainless_steel_shaft_6x103mm" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="shaft_collar_0_m0_{idx_magbot}" type="mesh" pos="0 0.0365 0.0334" euler="-1.5707963 0 1.5707963" '
                'mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="shaft_collar_1_m0_{idx_magbot}" type="mesh" pos="0 -0.0475 0.0334" euler="1.5707963 0 1.5707963" '
                'mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="toothed_wheel_m0_b1_{idx_magbot}" type="mesh" pos="0 0 0.008" euler="1.5707963 0 0" '
                'mesh="toothed_wheel_1" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="toothed_wheel_m0_b2_{idx_magbot}" type="mesh" pos="0 -0.025 0.0334" mesh="toothed_wheel_1" '
                'material="mat_tan"/>',
                f'\t\t\t\t\t<body name="HTD3M_pulley_m0_b_{idx_magbot}" pos="0 0.015 0.0334" euler="0 0 3.14159" gravcomp="1">',
                f'\t\t\t\t\t\t<geom name="HTD3M_pulley_m0_b_geom_{idx_magbot}" type="mesh" pos="0 0 0" mesh="HTD3M_pulley" '
                'material="light_gray"/>',
                f'\t\t\t\t\t\t<body name="k0_toothed_belt_{idx_magbot}" pos="0 0.0208 0" euler="0 -2.47259 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t<joint name="k0_toothed_belt_joint_{idx_magbot}" type="hinge" axis="0 -1 0" pos="0 0 0.0" damping="0"/>',
                f'\t\t\t\t\t\t\t<geom name="k0_toothed_belt_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" '
                'mesh="HTD3M_toothed_belt" material="black"/>',
                '\t\t\t\t\t\t</body>',
                '\t\t\t\t\t</body>',
                f'\t\t\t\t\t<body name="k0_connection_{idx_magbot}" pos="0.1021 0 0.0735" euler="0 -0.665 0" gravcomp="1">',
                f'\t\t\t\t\t\t<joint name="k0_connection_joint_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.091 0 0" damping="0" '
                'range="-0.3 0.6109"/>',
                f'\t\t\t\t\t\t<geom name="k0_connection_geom_{idx_magbot}" type="mesh" pos="0 0 0" mesh="solid_connection_m0m1" '
                'material="light_green"/>',
                f'\t\t\t\t\t\t<body name="platform_{idx_magbot}" pos="0.173 0 -0.003" euler="0 0.665 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t<joint name="connection_0_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.066 0 -0.0484" damping="0"/>',
                f'\t\t\t\t\t\t\t<geom name="platform_geom_{idx_magbot}" type="mesh" pos="0 0 0"  mass="0.01" mesh="platform" '
                'material="black"/>',
                f'\t\t\t\t\t\t\t<site name="platform_k1_site_left_{idx_magbot}" pos="0.066 0.016 -0.0484" size="0.002" type="sphere" '
                'rgba="0 1 0 0"/>',
                f'\t\t\t\t\t\t\t<site name="platform_k1_site_right_{idx_magbot}" pos="0.066 -0.016 -0.0484" size="0.002" type="sphere" '
                'rgba="0 1 0 0"/>',
                f'\t\t\t\t\t\t\t<site name="platform_k2_site_left_{idx_magbot}" pos="-0.016 -0.064 -0.0385" size="0.002" type="sphere" '
                'rgba="0 1 0 0"/>',
                f'\t\t\t\t\t\t\t<site name="platform_k2_site_right_{idx_magbot}" pos="0.016 -0.064 -0.0385" size="0.002" type="sphere" '
                'rgba="0 1 0 0"/>',
                '\t\t\t\t\t\t\t<!-- k0 -->',
                f'\t\t\t\t\t\t\t<geom name="stainless_steel_shaft_0_k0_{idx_magbot}" type="mesh" pos="-0.096 -0.0055 -0.032" '
                'mesh="stainless_steel_shaft_6x103mm" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="stainless_steel_shaft_1_k0_{idx_magbot}" type="mesh" pos="-0.096 -0.0055 0" '
                'mesh="stainless_steel_shaft_6x103mm" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_0_k0_{idx_magbot}" type="mesh" pos="-0.096 0.0365 -0.032" '
                'euler="-1.5707963 0 1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_1_k0_{idx_magbot}" type="mesh" pos="-0.096 -0.0475 -0.032" '
                'euler="1.5707963 0 1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_2_k0_{idx_magbot}" type="mesh" pos="-0.096 0.0365 0" '
                'euler="-1.5707963 0 1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_3_k0_{idx_magbot}" type="mesh" pos="-0.096 -0.0475 0" '
                'euler="1.5707963 0 1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_2_0_k0_{idx_magbot}" type="mesh" pos="-0.096 -0.0185 -0.032" '
                'euler="0 0 -1.5707963" mesh="toothed_wheel_2" material="mat_tan"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_2_1_k0_{idx_magbot}" type="mesh" pos="-0.096 -0.0185 0" euler="0 0 -1.5707963" '
                'mesh="toothed_wheel_2" material="mat_tan"/>',
                f'\t\t\t\t\t\t\t<geom name="HTD3M_pulley_k0_{idx_magbot}" type="mesh" pos="-0.096 0.015 -0.032" euler="0 0 3.14159" '
                'mesh="HTD3M_pulley" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_1_k0_{idx_magbot}" type="mesh" pos="-0.096 0.025 0" euler="0 0 3.14159" '
                'mesh="toothed_wheel_1" material="mat_tan"/>',
                '\t\t\t\t\t\t\t<!-- k1 -->',
                f'\t\t\t\t\t\t\t<geom name="stainless_steel_shaft_0_k1_{idx_magbot}" type="mesh" pos="0.096 -0.0055 -0.032" '
                'mesh="stainless_steel_shaft_6x103mm" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="stainless_steel_shaft_1_k1_{idx_magbot}" type="mesh" pos="0.096 -0.0055 0" '
                'mesh="stainless_steel_shaft_6x103mm" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_0_k1_{idx_magbot}" type="mesh" pos="0.096 0.0365 -0.032" '
                'euler="-1.5707963 0 -1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_1_k1_{idx_magbot}" type="mesh" pos="0.096 -0.0475 -0.032" '
                'euler="1.5707963 0 -1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_2_k1_{idx_magbot}" type="mesh" pos="0.096 0.0365 0" '
                'euler="-1.5707963 0 -1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_3_k1_{idx_magbot}" type="mesh" pos="0.096 -0.0475 0" '
                'euler="1.5707963 0 -1.5707963"  mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_2_0_k1_{idx_magbot}" type="mesh" pos="0.096 -0.0185 -0.032" '
                'euler="0 0 -1.5707963" mesh="toothed_wheel_2" material="mat_tan"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_2_1_k1_{idx_magbot}" type="mesh" pos="0.096 -0.0185 0" euler="0 0 -1.5707963" '
                'mesh="toothed_wheel_2" material="mat_tan"/>',
                f'\t\t\t\t\t\t\t<geom name="HTD3M_pulley_k1_{idx_magbot}" type="mesh" pos="0.096 0.015 -0.032" euler="0 0 3.14159" '
                'mesh="HTD3M_pulley" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_1_k1_{idx_magbot}" type="mesh" pos="0.096 0.025 0" euler="0 0 3.14159" '
                'mesh="toothed_wheel_1" material="mat_tan"/>',
                '\t\t\t\t\t\t\t<!-- k2 -->',
                f'\t\t\t\t\t\t\t<geom name="stainless_steel_shaft_0_k2_{idx_magbot}" type="mesh" pos="0 -0.096 -0.032" '
                'euler="0 0 1.5707963" mesh="stainless_steel_shaft_6x92mm" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="stainless_steel_shaft_1_k2_{idx_magbot}" type="mesh" pos="0 -0.096 0" euler="0 0 1.5707963" '
                'mesh="stainless_steel_shaft_6x92mm" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_0_k2_{idx_magbot}" type="mesh" pos="0.0365 -0.096 -0.032" '
                'euler="0 1.5707963 3.14159" mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_1_k2_{idx_magbot}" type="mesh" pos="-0.0365 -0.096 -0.032" '
                'euler="0 -1.5707963 3.14159" mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_2_k2_{idx_magbot}" type="mesh" pos="0.0365 -0.096 0" euler="0 1.5707963 3.14159" '
                'mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="shaft_collar_3_k2_{idx_magbot}" type="mesh" pos="-0.0365 -0.096 0" euler="0 -1.5707963 3.14159"'
                ' mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_2_0_k2_{idx_magbot}" type="mesh" pos="0.0075 -0.096 -0.032" '
                'mesh="toothed_wheel_2" material="mat_tan"/>',
                f'\t\t\t\t\t\t\t<geom name="toothed_wheel_2_1_k2_{idx_magbot}" type="mesh" pos="0.0075 -0.096 0" mesh="toothed_wheel_2" '
                'material="mat_tan"/>',
                f'\t\t\t\t\t\t\t<geom name="GT2_pulley_10mm_k2_{idx_magbot}" type="mesh" pos="-0.013 -0.096 -0.032" mesh="GT2_pulley_10mm"'
                ' material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="GT2_pulley_6mm_k2_{idx_magbot}" type="mesh" pos="-0.02 -0.096 0" mesh="GT2_pulley_6mm" '
                'material="light_gray"/>',
                f'\t\t\t\t\t\t\t<geom name="GT2_toothed_belt_6mm_k2_{idx_magbot}" type="mesh" pos="-0.0119 -0.096 0" '
                'mesh="GT2_toothed_belt_6mm" material="black"/>',
                f'\t\t\t\t\t\t\t<body name="gripper_mount_{idx_magbot}" pos="-0.46449586 -0.32982973 -0.215213" '
                'quat="0.99999508 0 0 -0.0031375702" gravcomp="1">',
                '\t\t\t\t\t\t\t\t<inertial pos="0 0 0" mass="1e-9" diaginertia="1e-9 1e-9 1e-9"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_wide_0_{idx_magbot}" type="mesh" pos="0.409518 0.332406 0.215213"'
                ' quat="0.707103 0.0022186 -0.707103 0.0022186" mesh="shaft_collar_wide" material="mat_tan" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_wide_1_{idx_magbot}" type="mesh" pos="0.530316 0.333164 0.215213"'
                ' quat="0.707103 0.0022186 -0.707103 0.0022186" mesh="shaft_collar_wide" material="mat_tan" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_0_{idx_magbot}" type="mesh" pos="0.558252 0.35934 0.215213" '
                'quat="0.707103 -0.707103 -0.0022186 0.0022186" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_1_{idx_magbot}" type="mesh" pos="0.558647 0.296342 0.183213"'
                ' quat="0.707103 0.707103 0.0022186 0.0022186" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_2_{idx_magbot}" type="mesh" pos="0.489019 0.236903 0.215213"'
                ' quat="0.707103 -0.0022186 0.707103 0.0022186" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_3_{idx_magbot}" type="mesh" pos="0.55871 0.286342 0.215213"'
                ' quat="0.707103 -0.707103 -0.0022186 0.0022186" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_4_{idx_magbot}" type="mesh" pos="0.558189 0.36934 0.183213"'
                ' quat="0.707103 0.707103 0.0022186 0.0022186" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_5_{idx_magbot}" type="mesh" pos="0.42702 0.236514 0.215213"'
                ' quat="0.707103 -0.0022186 0.707103 0.0022186" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_6_{idx_magbot}" type="mesh" pos="0.366193 0.368135 0.215213"'
                ' quat="0.498429 0.501566 -0.498429 0.501566" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_7_{idx_magbot}" type="mesh" pos="0.43702 0.236577 0.183213"'
                ' quat="0.707103 0.0022186 -0.707103 0.0022186" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_8_{idx_magbot}" type="mesh" pos="0.366193 0.368135 0.183213"'
                ' quat="0.498429 0.501566 -0.498429 0.501566" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_9_{idx_magbot}" type="mesh" pos="0.366651 0.295137 0.215213"'
                ' quat="0.498429 0.501566 -0.498429 0.501566" mesh="shaft_collar_thick" material="mat_tan" contype="0" conaffinity="0"'
                ' mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_10_{idx_magbot}" type="mesh" '
                'pos="0.366651 0.295137 0.183213" quat="0.498429 0.501566 -0.498429 0.501566" mesh="shaft_collar_thick" material="mat_tan"'
                ' contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thick_11_{idx_magbot}" type="mesh" '
                'pos="0.499019 0.236966 0.183213" quat="0.707103 0.0022186 -0.707103 0.0022186" mesh="shaft_collar_thick" '
                'material="mat_tan" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_with_hole_short_0_{idx_magbot}" type="mesh" '
                'pos="0.479817 0.332848 0.215213" quat="0.999995 5e-12 0 0.00313757" mesh="shaft_with_hole_short" material="light_gray" '
                'contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_with_hole_short_1_{idx_magbot}" type="mesh" '
                'pos="0.445017 0.332629 0.215213" quat="0 0.00313757 -0.999995 -5e-12" mesh="shaft_with_hole_short" material="light_gray" '
                'contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_toothed_wheel_large_0_{idx_magbot}" type="mesh" '
                'pos="0.505516 0.333009 0.215213" quat="4e-12 -0.709322 0.704885 4e-12" mesh="toothed_wheel_large" material="mat_tan" '
                'contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_toothed_wheel_large_1_{idx_magbot}" type="mesh" '
                'pos="0.419318 0.332468 0.215213" quat="0.709322 4e-12 4e-12 -0.704885" mesh="toothed_wheel_large" material="mat_tan" '
                'contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_mounting_bracket_0_{idx_magbot}" type="mesh" pos="0.413018 0.332428 0.215213" '
                'quat="0.999995 5e-12 0 0.00313757" mesh="mounting_bracket" material="black" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_mounting_bracket_1_{idx_magbot}" type="mesh" pos="0.511816 0.333048 0.215213" '
                'quat="0.999995 5e-12 0 0.00313757" mesh="mounting_bracket" material="black" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_flanged_shaft_0_{idx_magbot}" type="mesh" pos="0.417518 0.332457 0.215213" '
                'quat="0 0.00313757 -0.999995 -5e-12" mesh="flanged_shaft" material="black" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_flanged_shaft_1_{idx_magbot}" type="mesh" pos="0.507316 0.33302 0.215213" '
                'quat="0.999995 5e-12 0 0.00313757" mesh="flanged_shaft" material="black" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_6x92mm_0_{idx_magbot}" type="mesh" pos="0.463019 0.23674 0.183213" '
                'quat="0.709322 4e-12 4e-12 -0.704885" mesh="shaft_6x92mm" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_6x92mm_1_{idx_magbot}" type="mesh" pos="0.463019 0.23674 0.215213" '
                'quat="0.709322 4e-12 4e-12 -0.704885" mesh="shaft_6x92mm" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_6x103mm_0_{idx_magbot}" type="mesh" pos="0.366453 0.326636 0.215213" '
                'quat="0.999995 -0 0 0.00313757" mesh="shaft_6x103mm" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_6x103mm_1_{idx_magbot}" type="mesh" pos="0.366453 0.326636 0.183213" '
                'quat="0.999995 -0 0 0.00313757" mesh="shaft_6x103mm" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_6x103mm_2_{idx_magbot}" type="mesh" pos="0.55845 0.327841 0.183213" '
                'quat="0.999995 -0 0 0.00313757" mesh="shaft_6x103mm" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_6x103mm_3_{idx_magbot}" type="mesh" pos="0.55845 0.327841 0.215213" '
                'quat="0.999995 -0 0 0.00313757" mesh="shaft_6x103mm" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thin_0_{idx_magbot}" type="mesh" pos="0.445017 0.332629 0.215213" '
                'quat="0.707103 0.0022186 -0.707103 0.0022186" mesh="shaft_collar_thin" material="light_gray" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_collar_thin_1_{idx_magbot}" type="mesh" pos="0.484817 0.332879 0.215213" '
                'quat="0.707103 0.0022186 -0.707103 0.0022186" mesh="shaft_collar_thin" material="light_gray" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_toothed_wheel_1_0_{idx_magbot}" type="mesh" pos="0.558258 0.358393 0.215213" '
                'quat="1.67e-10 -0.999995 -0.00313757 -5.3271e-08" mesh="toothed_wheel_1" material="mat_tan" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_toothed_wheel_1_1_{idx_magbot}" type="mesh" pos="0.391518 0.332293 0.215213" '
                'quat="4e-12 -0.709322 0.704885 4e-12" mesh="toothed_wheel_1" material="mat_tan" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_toothed_wheel_1_2_{idx_magbot}" type="mesh" pos="0.366262 0.357188 0.215213" '
                'quat="1.67e-10 -0.999995 -0.00313757 -5.3271e-08" mesh="toothed_wheel_1" material="mat_tan" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_toothed_wheel_1_3_{idx_magbot}" type="mesh" pos="0.533316 0.333183 0.215213" '
                'quat="0.709322 1.18e-10 1.19e-10 -0.704885" mesh="toothed_wheel_1" material="mat_tan" contype="0" conaffinity="0" '
                'mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_flanged_pulley_{idx_magbot}" type="mesh" pos="0.448517 0.332651 0.215213" '
                'quat="0.999995 5e-12 0 0.00313757" mesh="flanged_pulley" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_shaft_3x45mm_{idx_magbot}" type="mesh" pos="0.462317 0.332738 0.215213" '
                'quat="0.999995 0 -0 0.00313757" mesh="shaft_3x45mm" material="light_gray" contype="0" conaffinity="0" mass="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_mount_toothed_wheel_small_{idx_magbot}" type="mesh" pos="0.473317 0.332807 0.215213" '
                'quat="0 -0.709322 0.704885 -0" mesh="toothed_wheel_small" material="mat_tan" contype="0" conaffinity="0" mass="0"/>',
                '\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t<!-- gripper end-effector (mounted on the platform) -->',
                f'\t\t\t\t\t\t\t<body name="gripper_platform_{idx_magbot}" pos="-0.46449586 -0.32982973 -0.215213" '
                'quat="0.99999508 0 0 -0.0031375702" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t<joint name="gripper_alpha_joint_{idx_magbot}" type="hinge" axis="1 0 0" pos="0.462517 0.33276 0.215213" '
                'range="-3.14159 0"/>',
                '\t\t\t\t\t\t\t\t<inertial pos="0.462517 0.332718 0.254927" mass="0.08" fullinertia="3e-5 3e-5 5e-5 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_platform_geom_{idx_magbot}" type="mesh" pos="0.462517 0.33276 0.258213" '
                'quat="0.999995 -0.000247894 -7.7779e-07 0.00313757" mesh="gripper_platform" material="black" contype="128" '
                'conaffinity="383"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_platform_toothed_wheel_large_{idx_magbot}" type="mesh" '
                'pos="0.462517 0.33276 0.258213" quat="0.497674 -0.497922 -0.502069 0.502316" mesh="toothed_wheel_large" '
                'material="mat_tan" contype="128" conaffinity="383"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_platform_shaft_with_hole_long_{idx_magbot}" type="mesh" '
                'pos="0.462517 0.332748 0.232513" quat="0.99999 -0.000247895 -4.66135e-07 0.00439477" mesh="shaft_with_hole_long" '
                'material="light_gray" contype="128" conaffinity="383"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_platform_shaft_collar_wide_{idx_magbot}" type="mesh" pos="0.462517 0.33276 0.258213"'
                ' quat="0.00439477 4.66135e-07 -0.000247895 -0.99999" mesh="shaft_collar_wide" material="mat_tan" contype="128" '
                'conaffinity="383"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_platform_shaft_collar_thin_{idx_magbot}" type="mesh" pos="0.462517 0.332748 0.232513"'
                ' quat="0.99999 -0.000247895 -4.66135e-07 0.00439477" mesh="shaft_collar_thin" material="light_gray" contype="128" '
                'conaffinity="383"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_platform_shaft_3x153mm_{idx_magbot}" type="mesh" pos="0.462517 0.33274 0.217013" '
                'quat="0.983754 -0.000243584 -4.60323e-05 -0.179523" mesh="shaft_3x153mm" material="light_gray" contype="128" '
                'conaffinity="383"/>',
                f'\t\t\t\t\t\t\t\t<geom name="gripper_platform_toothed_wheel_small_{idx_magbot}" type="mesh" '
                'pos="0.462517 0.332744 0.226057" quat="0.695447 -0.695791 0.126909 -0.126974" mesh="toothed_wheel_small" '
                'material="mat_tan" contype="128" conaffinity="383"/>',
                '\t\t\t\t\t\t\t\t<!-- gripper assembly -->',
                f'\t\t\t\t\t\t\t\t<body name="gripper_assembly_{idx_magbot}" pos="0.32143033 -0.21927781 0" '
                'quat="0.93990062 0 0 0.34144814" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t<joint name="gripper_beta_joint_{idx_magbot}" type="hinge" axis="0 0 1" '
                'pos="0.462517 0.33276 0.332013"/>',
                '\t\t\t\t\t\t\t\t\t<inertial pos="0.462511 0.332807 0.348822" mass="0.06" fullinertia="2e-5 2e-5 3e-5 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t<geom name="gripper_assembly_shaft_collar_thin_{idx_magbot}" type="mesh" '
                'pos="0.462517 0.3328 0.337513" quat="0.99999 -0.000247895 -4.66135e-07 0.00439477" mesh="shaft_collar_thin" '
                'material="light_gray" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t<body name="gripper_assembly_base_{idx_magbot}" pos="0.462517 0.332797 0.332013" '
                'quat="0.939868 -0.000232453 -8.61261e-05 -0.341538" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t<geom name="gripper_assembly_base_{idx_magbot}" type="mesh" pos="0 0 0" mesh="gripper_base" '
                'material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t\t<site name="gripper_pose_site_{idx_magbot}" pos="0 0 0.0762" size="0.002" type="sphere" '
                'rgba="0 1 0 0"/>',
                '\t\t\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t\t\t<geom name="gripper_assembly_pivot_link_{idx_magbot}" type="mesh" pos="0.462517 0.332811 0.361383" '
                'quat="0.939868 -0.000232453 -8.61261e-05 -0.341538" mesh="pivot_link" material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t<geom name="gripper_assembly_shaft_collar_small_{idx_magbot}" type="mesh" '
                'pos="0.462517 0.332803 0.345013" quat="0.439079 0.554266 0.438803 -0.554482" mesh="shaft_collar_small" '
                'material="light_gray" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t<body name="gripper_jaw_1_{idx_magbot}" pos="0.449732 0.330475 0.369017" '
                'quat="0.860266 0.327247 0.390963 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t<joint name="gripper_jaw_1_hinge_{idx_magbot}" type="hinge" axis="0.642 0.767 0" pos="0 0 0" '
                'frictionloss="0.0" armature="0.005"/>',
                '\t\t\t\t\t\t\t\t\t\t<inertial pos="0 0 0" mass="0.01" fullinertia="2e-6 2e-6 2e-6 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_1_link_0_{idx_magbot}" type="mesh" pos="-0.002790 0.002340 0.003427" '
                'quat="0.34655 -0.317547 -0.873644 -0.125746" mesh="jaw_link" material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_1_link_1_{idx_magbot}" type="mesh" pos="0.010050 0.017674 0.003419" '
                'quat="0.34655 -0.317547 -0.873644 -0.125746" mesh="jaw_link" material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t\t<body name="gripper_jaw_1_finger_{idx_magbot}" pos="-0.005581 0.004677 0.006854" '
                'quat="0.871064 -0.315259 -0.376641 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t\t<joint name="gripper_jaw_1_finger_hinge_{idx_magbot}" type="hinge" axis="0.642 0.767 0" pos="0 0 0"'
                ' frictionloss="0" armature="0"/>',
                '\t\t\t\t\t\t\t\t\t\t\t<inertial pos="0 0 0" mass="0.02" fullinertia="5e-6 5e-6 5e-6 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_1_finger_coupler_{idx_magbot}" type="mesh" '
                'pos="0.013105 0.002066 -0.011493" quat="0.679744 -0.236013 -0.649076 -0.246873" mesh="finger_coupler" material="black" '
                'contype="0" conaffinity="0"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_1_finger_body_{idx_magbot}" type="mesh" pos="0.000753 0.012414 -0.001276" '
                'quat="0.936468 -0.0292271 -0.0798775 -0.340285" mesh="finger_body" material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_pad_1_{idx_magbot}" type="mesh" pos="0.008070 0.006291 0.007197" '
                'quat="0.0085903 0.341522 0.939829 -0.00332277" mesh="finger_pad" material="mat_amber" contype="256" conaffinity="255" '
                'friction="3.0 0.1 0.01" condim="4" priority="3"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<body name="gripper_jaw_1_arm_B_{idx_magbot}" pos="-0.001687 0.009499 -0.001274" '
                'quat="0.930795 0.234624 0.280307 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<joint name="gripper_jaw_1_arm_B_hinge_{idx_magbot}" type="hinge" axis="0.642 0.767 0" '
                'pos="0 0 0" frictionloss="0" armature="0"/>',
                '\t\t\t\t\t\t\t\t\t\t\t\t<inertial pos="0 0 0" mass="0.005" fullinertia="1e-6 1e-6 1e-6 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_1_four_bar_link_{idx_magbot}" type="mesh" '
                'pos="0.014899 -0.007521 -0.007223" quat="0.924583 -0.0615422 -0.168811 -0.335948" mesh="four_bar_link" material="black" '
                'contype="0" conaffinity="0"/>',
                '\t\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t\t\t<body name="gripper_jaw_2_{idx_magbot}" pos="0.462460 0.319821 0.369022" '
                'quat="0.860263 -0.327249 -0.390966 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t<joint name="gripper_jaw_2_hinge_{idx_magbot}" type="hinge" axis="0.642 0.767 0" pos="0 0 0" '
                'frictionloss="0" armature="0.005"/>',
                '\t\t\t\t\t\t\t\t\t\t<inertial pos="0 0 0" mass="0.01" fullinertia="2e-6 2e-6 2e-6 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_2_link_0_{idx_magbot}" type="mesh" pos="0.002791 -0.002336 0.003429" '
                'quat="0.873581 -0.126178 -0.34671 -0.317376" mesh="jaw_link" material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_2_link_1_{idx_magbot}" type="mesh" pos="0.015631 0.012998 0.003422" '
                'quat="0.873581 -0.126178 -0.34671 -0.317376" mesh="jaw_link" material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t\t<body name="gripper_jaw_2_finger_{idx_magbot}" pos="0.005582 -0.004672 0.006858" '
                'quat="0.870962 0.315375 0.376780 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t\t<joint name="gripper_jaw_2_finger_hinge_{idx_magbot}" type="hinge" axis="0.642 0.767 0" '
                'pos="0 0 0" frictionloss="0" armature="0"/>',
                '\t\t\t\t\t\t\t\t\t\t\t<inertial pos="0 0 0" mass="0.02" fullinertia="5e-6 5e-6 5e-6 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_2_finger_coupler_{idx_magbot}" type="mesh" '
                'pos="-0.000265 0.013259 -0.011498" quat="0.648951 -0.247194 -0.679863 -0.235676" mesh="finger_coupler" material="black" '
                'contype="0" conaffinity="0"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_2_finger_body_{idx_magbot}" type="mesh" pos="0.012087 0.002921 -0.001270" '
                'quat="0.340325 0.0797059 -0.0287638 0.936482" mesh="finger_body" material="black" contype="256" conaffinity="255"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_pad_2_{idx_magbot}" type="mesh" pos="0.004770 0.009053 0.007196" '
                'quat="0.00285788 -0.93983 0.341526 0.00841805" mesh="finger_pad" material="mat_amber" contype="256" conaffinity="255" '
                'friction="3.0 0.1 0.01" condim="4" priority="3"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<body name="gripper_jaw_2_arm_B_{idx_magbot}" pos="0.009646 0.000008 -0.001266" '
                'quat="0.930740 -0.234715 -0.280414 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<joint name="gripper_jaw_2_arm_B_hinge_{idx_magbot}" type="hinge" axis="0.642 0.767 0" '
                'pos="0 0 0" frictionloss="0" armature="0"/>',
                '\t\t\t\t\t\t\t\t\t\t\t\t<inertial pos="0 0 0" mass="0.005" fullinertia="1e-6 1e-6 1e-6 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<geom name="gripper_jaw_2_four_bar_link_{idx_magbot}" type="mesh" '
                'pos="-0.010018 0.013342 -0.007235" quat="0.336031 0.168642 -0.0610848 0.924614" mesh="four_bar_link" material="black" '
                'contype="0" conaffinity="0"/>',
                '\t\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t</body>',
                '\t\t\t\t\t</body>',
                '\t\t\t\t</body>',
                '\t\t\t</body>',
                '\t\t\t<!-- connection 1 -->',
                f'\t\t\t<body name="rail_mount_1_{idx_magbot}" pos="0.48 0 0" euler="0 0 3.14159" gravcomp="1">',
                f'\t\t\t\t<joint name="rail_mount_1_joint_{idx_magbot}" type="slide" axis="1 0 0" damping="0.1"/>',
                f'\t\t\t\t<joint name="rail_mount_1_joint_yaw_{idx_magbot}" type="hinge" axis="0 0 1"/>',
                f'\t\t\t\t<geom name="rail_mount_geom_1_{idx_magbot}" type="mesh" euler="0 0 -1.5707963" mesh="rail_mount_m1m2_mesh" '
                'mass="0.5" pos="0 0 0" material="black"/>',
                f'\t\t\t\t<site name="rail_mount_1_site_{idx_magbot}" pos="0 0 0" size="0.001" type="sphere"/>',
                f'\t\t\t\t<body name="rotation_plate_m1_{idx_magbot}" pos="0 0 0.0285" euler="0 0 3.14159" gravcomp="1">',
                f'\t\t\t\t\t<joint name="rotation_plate_m1_joint_{idx_magbot}" type="hinge" axis="0 0 1" pos="0 0 0" damping="0"/>',
                f'\t\t\t\t\t<geom name="rotation_plate_m1_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" mesh="rotation_plate_m1" '
                'material="black"/>',
                f'\t\t\t\t\t<geom name="slide_bearing_0_m1_{idx_magbot}" type="mesh" pos="0 0.036 0.0334" euler="1.5707963 0 0" '
                'mesh="slide_bearing" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="slide_bearing_1_m1_{idx_magbot}" type="mesh" pos="0 -0.047 0.0334" euler="-1.5707963 0 0" '
                'mesh="slide_bearing" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="stainless_steel_shaft_m1_b_{idx_magbot}" type="mesh" pos="0 -0.0055 0.0334" '
                'mesh="stainless_steel_shaft_6x103mm" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="shaft_collar_0_m1_{idx_magbot}" type="mesh" pos="0 0.0365 0.0334" euler="-1.5707963 0 -1.5707963" '
                'mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="shaft_collar_1_m1_{idx_magbot}" type="mesh" pos="0 -0.0475 0.0334" euler="1.5707963 0 -1.5707963" '
                'mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="toothed_wheel_m1_b1_{idx_magbot}" type="mesh" pos="0 0 0.008" euler="1.5707963 0 0" '
                'mesh="toothed_wheel_1" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="toothed_wheel_m1_b2_{idx_magbot}" type="mesh" pos="0 -0.025 0.0334" mesh="toothed_wheel_1" '
                'material="mat_tan"/>',
                f'\t\t\t\t\t<body name="HTD3M_pulley_m1_b_{idx_magbot}" pos="0 0.015 0.0334" euler="0 0 3.14159" gravcomp="1">',
                f'\t\t\t\t\t\t<geom name="HTD3M_pulley_m1_b_geom_{idx_magbot}" type="mesh" pos="0 0 0" mesh="HTD3M_pulley" '
                'material="light_gray"/>',
                f'\t\t\t\t\t\t<body name="k1_toothed_belt_{idx_magbot}" pos="0 0.0208 0" euler="0 -0.662 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t<geom name="k1_toothed_belt_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" '
                'mesh="HTD3M_toothed_belt" material="black"/>',
                f'\t\t\t\t\t\t\t<joint name="k1_toothed_belt_joint_{idx_magbot}" type="hinge" axis="0 1 0" pos="0 0 0.0" damping="0"/>',
                '\t\t\t\t\t\t</body>',
                '\t\t\t\t\t</body>',
                f'\t\t\t\t\t<body name="k1_connection_{idx_magbot}" pos="-0.1021 0 0.0735" euler="0 0.665 3.14159" gravcomp="1">',
                f'\t\t\t\t\t\t<geom name="k1_connection_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" mesh="solid_connection_m0m1"'
                ' material="light_green"/>',
                f'\t\t\t\t\t\t<joint name="k1_connection_joint_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.091 0 0" damping="0" '
                'range="-0.3 0.6109"/>',
                f'\t\t\t\t\t\t<site name="k1_connection_site_left_{idx_magbot}" pos="0.091 -0.016 0" size="0.002" type="sphere" '
                'rgba="1 0 0 0"/>',
                f'\t\t\t\t\t\t<site name="k1_connection_site_right_{idx_magbot}" pos="0.091 0.016 0" size="0.002" type="sphere" '
                'rgba="1 0 0 0"/>',
                '\t\t\t\t\t</body>',
                '\t\t\t\t</body>',
                '\t\t\t</body>',
                '\t\t\t<!-- connection 2 -->',
                f'\t\t\t<body name="rail_mount_2_{idx_magbot}" pos="0.24 -0.12 0" gravcomp="1">',
                f'\t\t\t\t<joint name="rail_mount_2_joint_{idx_magbot}" type="slide" axis="0 1 0" damping="0.1"/>',
                f'\t\t\t\t<joint name="rail_mount_2_joint_yaw_{idx_magbot}" type="hinge" axis="0 0 1"/>',
                f'\t\t\t\t<geom name="rail_mount_geom_2_{idx_magbot}" type="mesh" mesh="rail_mount_m1m2_mesh" euler="0 0 1.5707963" '
                'mass="0.5" pos="0 0 0" material="black"/>',
                f'\t\t\t\t<site name="rail_mount_2_site_{idx_magbot}" pos="0 0 0" size="0.001" type="sphere"/>',
                f'\t\t\t\t<body name="rotation_plate_m2_{idx_magbot}" pos="0 0 0.0285" gravcomp="1">',
                f'\t\t\t\t\t<joint name="rotation_plate_m2_joint_{idx_magbot}" type="hinge" axis="0 0 1" pos="0 0 0" damping="0"/>',
                f'\t\t\t\t\t<geom name="rotation_plate_m2_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" mesh="rotation_plate_m2" '
                'material="black"/>',
                f'\t\t\t\t\t<geom name="slide_bearing_0_m2_{idx_magbot}" type="mesh" pos="0.036 0 0.0334" euler="0 -1.5707963 0"  '
                'mesh="slide_bearing" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="slide_bearing_1_m2_{idx_magbot}" type="mesh" pos="-0.036 0 0.0334" euler="0 1.5707963 0" '
                'mesh="slide_bearing" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="stainless_steel_shaft_m2_b_{idx_magbot}" type="mesh" pos="0 0 0.0334" euler="0 0 1.5707963" '
                'mesh="stainless_steel_shaft_6x92mm" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="shaft_collar_0_m2_{idx_magbot}" type="mesh" pos="0.0365 0 0.0334" euler="0 1.5707963 3.14159" '
                'mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="shaft_collar_1_m2_{idx_magbot}" type="mesh" pos="-0.0365 0 0.0334" euler="0 -1.5707963 3.14159" '
                'mesh="shaft_collar_large" material="light_gray"/>',
                f'\t\t\t\t\t<geom name="toothed_wheel_m2_b1_{idx_magbot}" type="mesh" pos="0 0 0.008" euler="1.5707963 0 0" '
                'mesh="toothed_wheel_1" material="mat_tan"/>',
                f'\t\t\t\t\t<geom name="toothed_wheel_m2_b2_{idx_magbot}" type="mesh" pos="0.025 0 0.0334" euler="0 0 1.5707963" '
                'mesh="toothed_wheel_1" material="mat_tan"/>',
                f'\t\t\t\t\t<body name="GT2_pulley_10mm_m2_b_{idx_magbot}" pos="-0.013 0 0.0334" gravcomp="1">',
                f'\t\t\t\t\t\t<geom name="GT2_pulley_10mm_m2_b_geom_{idx_magbot}" type="mesh" pos="0 0 0" mesh="GT2_pulley_10mm" '
                'material="light_gray"/>',
                f'\t\t\t\t\t\t<body name="k2_toothed_belt_{idx_magbot}" pos="0.0079 0 0" euler="1.365 0 0" gravcomp="1">',
                f'\t\t\t\t\t\t\t<geom name="k2_toothed_belt_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" '
                'mesh="GT2_toothed_belt_10mm" material="black"/>',
                f'\t\t\t\t\t\t\t<joint name="k2_toothed_belt_joint_{idx_magbot}" type="hinge" axis="-1 0 0" pos="0 0 0.0" damping="0"/>',
                '\t\t\t\t\t\t</body>',
                '\t\t\t\t\t</body>',
                f'\t\t\t\t\t<body name="k2_connection_{idx_magbot}" pos="0 0.044 0.0833" euler="1.355 0 0" gravcomp="1">',
                f'\t\t\t\t\t\t<geom name="k2_connection_geom_{idx_magbot}" type="mesh" pos="0 0 0" mass="0.01" mesh="solid_connection_m2" '
                'material="light_green"/>',
                f'\t\t\t\t\t\t<joint name="k2_connection_joint_{idx_magbot}" type="hinge" axis="-1 0 0" pos="0 -0.057 0" damping="0"/>',
                f'\t\t\t\t\t\t<site name="k2_connection_site_left_{idx_magbot}" pos="-0.016 0.057 0" size="0.002" type="sphere" '
                'rgba="1 0 0 0"/>',
                f'\t\t\t\t\t\t<site name="k2_connection_site_right_{idx_magbot}" pos="0.016 0.057 0" size="0.002" type="sphere" '
                'rgba="1 0 0 0"/>',
                '\t\t\t\t\t</body>',
                '\t\t\t\t</body>',
                '\t\t\t</body>',
                '\t\t</body>',
            ]
            if idx_magbot == 0:
                gripper_magbot_list.append(
                    '\t\t<site name="world_ref" pos="0 0 0" euler="0 0 0" rgba="0 0 0 0"/>',
                )
            gripper_magbot_xml_str = '\n'.join(gripper_magbot_list)
            custom_worldbody_xml_str = custom_model_xml_strings.get('custom_worldbody_xml_str', None)
            if custom_worldbody_xml_str is not None:
                custom_worldbody_xml_str += gripper_magbot_xml_str
            else:
                custom_worldbody_xml_str = gripper_magbot_xml_str
            custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

            # contacts
            gripper_contact_list = [
                f'\n\t\t<!-- Gripper MagBot {idx_magbot} -->',
                f'\t\t<exclude body1="rail_mount_0_{idx_magbot}" body2="mover_{self.indices_mover_c0[idx_magbot]}"/>',
                f'\t\t<exclude body1="rail_mount_0_{idx_magbot}" body2="m_{self.indices_mover_c0[idx_magbot]}_rail"/>',
                f'\t\t<exclude body1="rail_mount_0_{idx_magbot}" body2="k0_connection_{idx_magbot}"/>',
                f'\t\t<exclude body1="rail_mount_1_{idx_magbot}" body2="mover_{self.indices_mover_c1[idx_magbot]}"/>',
                f'\t\t<exclude body1="rail_mount_1_{idx_magbot}" body2="m_{self.indices_mover_c1[idx_magbot]}_rail"/>',
                f'\t\t<exclude body1="rail_mount_1_{idx_magbot}" body2="k1_connection_{idx_magbot}"/>',
                f'\t\t<exclude body1="rail_mount_2_{idx_magbot}" body2="mover_{self.indices_mover_c2[idx_magbot]}"/>',
                f'\t\t<exclude body1="rail_mount_2_{idx_magbot}" body2="m_{self.indices_mover_c2[idx_magbot]}_rail"/>',
                f'\t\t<exclude body1="rail_mount_2_{idx_magbot}" body2="k2_connection_{idx_magbot}"/>',
                f'\t\t<exclude body1="platform_{idx_magbot}" body2="k1_connection_{idx_magbot}"/>',
                f'\t\t<exclude body1="platform_{idx_magbot}" body2="k2_connection_{idx_magbot}"/>',
                f'\t\t<exclude body1="k0_toothed_belt_{idx_magbot}" body2="platform_{idx_magbot}"/>',
                f'\t\t<exclude body1="k1_toothed_belt_{idx_magbot}" body2="platform_{idx_magbot}"/>',
                f'\t\t<exclude body1="k2_toothed_belt_{idx_magbot}" body2="platform_{idx_magbot}"/>',
            ]
            gripper_contact_xml_str = '\n'.join(gripper_contact_list)
            custom_contact_xml_str = custom_model_xml_strings.get('custom_contact_xml_str', None)
            if custom_contact_xml_str is not None:
                custom_contact_xml_str += gripper_contact_xml_str
            else:
                custom_contact_xml_str = gripper_contact_xml_str
            custom_model_xml_strings['custom_contact_xml_str'] = custom_contact_xml_str

            # equality constraints
            gripper_equality_list = [
                '\n\t<equality>',
                f'\t\t<weld site1="m_{self.indices_mover_c0[idx_magbot]}_rail_site" site2="rail_mount_0_site_{idx_magbot}"/>',
                f'\t\t<weld site1="m_{self.indices_mover_c1[idx_magbot]}_rail_site" site2="rail_mount_1_site_{idx_magbot}"/>',
                f'\t\t<weld site1="m_{self.indices_mover_c2[idx_magbot]}_rail_site" site2="rail_mount_2_site_{idx_magbot}"/>',
                f'\t\t<joint joint1="k0_connection_joint_{idx_magbot}" joint2="k1_connection_joint_{idx_magbot}"/>',
                f'\t\t<joint joint1="k0_connection_joint_{idx_magbot}" joint2="k0_toothed_belt_joint_{idx_magbot}"/>',
                f'\t\t<joint joint1="k1_connection_joint_{idx_magbot}" joint2="k1_toothed_belt_joint_{idx_magbot}"/>',
                f'\t\t<joint joint1="k2_connection_joint_{idx_magbot}" joint2="k2_toothed_belt_joint_{idx_magbot}"/>',
                f'\t\t<connect site1="k1_connection_site_left_{idx_magbot}" site2="platform_k1_site_left_{idx_magbot}" solref="0.005 1" '
                'solimp="0.9 0.95 0.001"/>',
                f'\t\t<connect site1="k1_connection_site_right_{idx_magbot}" site2="platform_k1_site_right_{idx_magbot}" solref="0.005 1" '
                'solimp="0.9 0.95 0.001"/>',
                f'\t\t<connect site1="k2_connection_site_left_{idx_magbot}" site2="platform_k2_site_left_{idx_magbot}" solref="0.005 1" '
                'solimp="0.9 0.95 0.001"/>',
                f'\t\t<connect site1="k2_connection_site_right_{idx_magbot}" site2="platform_k2_site_right_{idx_magbot}" solref="0.005 1" '
                'solimp="0.9 0.95 0.001"/>',
                '\t\t<!-- gripper end-effector four-bar linkages and jaw coupling -->',
                f'\t\t<connect name="jaw1_four_bar_close_{idx_magbot}" body1="gripper_jaw_1_arm_B_{idx_magbot}" '
                f'body2="gripper_assembly_{idx_magbot}" anchor="0.005364 -0.007106 -0.010083"/>',
                f'\t\t<joint name="gripper_jaw_coupling_{idx_magbot}" joint1="gripper_jaw_1_hinge_{idx_magbot}" '
                f'joint2="gripper_jaw_2_hinge_{idx_magbot}" polycoef="0 -1 0 0 0"/>',
                f'\t\t<connect name="jaw2_four_bar_close_{idx_magbot}" body1="gripper_jaw_2_arm_B_{idx_magbot}" '
                f'body2="gripper_assembly_{idx_magbot}" anchor="-0.007931 0.004029 -0.010092"/>',
                '\t</equality>',
            ]
            gripper_equality_xml_str = '\n'.join(gripper_equality_list)
            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            if custom_outworldbody_xml_str is not None:
                custom_outworldbody_xml_str += gripper_equality_xml_str
            else:
                custom_outworldbody_xml_str = gripper_equality_xml_str
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        return custom_model_xml_strings

    def generate_actuator_xml_strings(self) -> str:
        """Generate MuJoCo XML strings for the c-rot actuators of the movers, gripper's a,b-rot actuators, and finger actuators of all
        Gripper MagBots.

        :return: the MuJoCo XML actuator string (start '<actuator>' and end '</actuator>' are not included)
        """
        self.gripper_abf_actuator_names = []
        self.mover_actuator_names = []
        actuator_str_list = []

        # mover actuators
        for idx_mover in self.mover_indices.flatten():
            site_name = f'site_m_{idx_mover}'
            current_actuator_name = f'mover_actuator_c_{idx_mover}'
            self.mover_actuator_names.append(current_actuator_name)
            actuator_str_list.append(f'\n\t\t<!-- actuators mover {idx_mover} -->')
            actuator_str_list.append(
                f'\t\t<general name="{current_actuator_name}" site="{site_name}" gear="0 0 0 0 0 1" ctrlrange="-100 100" '
                'refsite="world_ref"/>'
            )
            actuator_str_list.append('\n')

        # gripper actuators
        for idx_magbot in range(0, self.num_magbots):
            current_actuator_names = [
                f'gripper_actuator_a_{idx_magbot}',
                f'gripper_actuator_b_{idx_magbot}',
                f'gripper_actuator_f_{idx_magbot}',
            ]
            self.gripper_abf_actuator_names.append(current_actuator_names)
            actuator_str_list.append(f'\n\t\t<!-- gripper A,B,F actuators Gripper MagBot {idx_magbot} -->')
            actuator_str_list.append(
                f'\t\t<position name="{current_actuator_names[0]}" joint="gripper_alpha_joint_{idx_magbot}" kp="100" dampratio="1" '
                'ctrlrange="-3.14159 0"/>'
            )
            actuator_str_list.append(
                f'\t\t<position name="{current_actuator_names[1]}" joint="gripper_beta_joint_{idx_magbot}" kp="100" dampratio="1"/>'
            )
            actuator_str_list.append(
                f'\t\t<position name="{current_actuator_names[2]}" joint="gripper_jaw_1_hinge_{idx_magbot}" gear="-1" kp="100" '
                'dampratio="1" ctrlrange="0 1.07"/>'
            )

        actuator_xml_str = '\n'.join(actuator_str_list)
        return actuator_xml_str

    def _get_desired_mover_distances(
        self, z_d: np.ndarray, a_d: np.ndarray, z_m: np.ndarray, single_track: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the desired distances between the movers based on the desired positions and orientation of the gripper.
        This method is vectorized and can therefore calculate desired mover distances for multiple MagBots.

        :param z_d: the desired z-position of the gripper. Shape: (num_magbots,)
        :param a_d: the desired a-rotation of the gripper. Shape: (num_magbots,)
        :param z_m: the desired mover flight altitude. Shape: (num_magbots,)
        :param single_track: whether to calculate the desired mover distances in single-track mode. Shape: (num_magbots,)
        :return: the desired distances between the movers, i.e. two numpy arrays of shape (num_magbots,):
            - d_m01: distance between the mover attached to connection 0 and the mover attached to connection 1
            - d_m2p: distance between the mover attached to connection 2 and the platform (projection to x,y-plane)
        """
        # platform origin height
        z_p = np.minimum(np.maximum(z_d - z_m - self.magbot_params.n * np.cos(a_d), self.magbot_params.z_p_min), self.magbot_params.z_p_max)
        # distance between mover 0 and mover 1
        d_m01 = 2 * (
            np.sqrt(np.power(self.magbot_params.k_01, 2) - np.power(z_p - self.magbot_params.z_b01 - self.magbot_params.z_t01, 2))
            + self.magbot_params.x_b01
            + self.magbot_params.x_t01
        )
        # distance between platform origin and mover 2
        sign = 1 - 2 * single_track
        d_m2p = np.maximum(
            self.magbot_params.x_t2
            + self.magbot_params.x_b2
            + sign * np.sqrt(np.power(self.magbot_params.k_2, 2) - np.power(z_p - self.magbot_params.z_t2 - self.magbot_params.z_b2, 2)),
            0,
        )

        return d_m01, d_m2p

    def _get_desired_mover_xy_positions(
        self, x_d: np.ndarray, y_d: np.ndarray, a_d: np.ndarray, c_d: np.ndarray, d_m01: np.ndarray, d_m2p: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the desired x- and y-positions of the three movers based on the desired
        position and orientation of the gripper. This method is vectorized and can therefore
        calculate desired mover rotations for multiple MagBots.

        :param x_d: the desired x-position of the gripper. Shape: (num_magbots,)
        :param y_d: the desired y-position of the gripper. Shape: (num_magbots,)
        :param a_d: the desired a-rotation of the gripper. Shape: (num_magbots,)
        :param c_d: the desired c-rotation of the gripper. Shape: (num_magbots,)
        :param d_m01: the desired distance between mover 0 and mover 1. Shape: (num_magbots,)
        :param d_m2p: the desired distance between mover 2 and the platform. Shape: (num_magbots,)
        :return: the desired x- and y-positions of the movers, i.e. three numpy arrays of shape (num_magbots,):
            - x_m0: x-positions for movers attached to connection 0
            - y_m0: y-positions for movers attached to connection 0
            - x_m1: x-positions for movers attached to connection 1
            - y_m1: y-positions for movers attached to connection 1
            - x_m2: x-positions for movers attached to connection 2
            - y_m2: y-positions for movers attached to connection 2
        """
        # desired platform xy-pos
        x_p = x_d + self.magbot_params.n * np.abs(np.sin(a_d)) * np.sin(c_d)
        y_p = y_d - self.magbot_params.n * np.abs(np.sin(a_d)) * np.cos(c_d)

        # desired mover xy-pos
        # mover 0
        x_m0 = x_p - np.divide(d_m01, 2) * np.cos(c_d)
        y_m0 = y_p - np.divide(d_m01, 2) * np.sin(c_d)
        # mover 1
        x_m1 = x_p + np.divide(d_m01, 2) * np.cos(c_d)
        y_m1 = y_p + np.divide(d_m01, 2) * np.sin(c_d)
        # mover 2
        x_m2 = x_p + d_m2p * np.sin(c_d)
        y_m2 = y_p - d_m2p * np.cos(c_d)

        return x_m0, y_m0, x_m1, y_m1, x_m2, y_m2

    def _get_desired_mover_c_rotations(
        self, a_d: np.ndarray, b_d: np.ndarray, c_d: np.ndarray, f_d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the desired c-rotations of the three movers based on the desired rotations of the gripper, and
        the opening of the fingers. This method is vectorized and can therefore calculate desired mover rotations
        for multiple MagBots.

        :param a_d: the desired a-rotation of the gripper. Shape: (num_magbots,)
        :param b_d: the desired b-rotation of the gripper. Shape: (num_magbots,)
        :param c_d: the desired c-rotation of the gripper. Shape: (num_magbots,)
        :param f_d: the desired distance between the fingers. Shape: (num_magbots,)
        :return: the desired c-rotations of all three movers, i.e. three numpy array of shape (num_magbots,):
            - c_m0: c-rotations for movers attached to connection 0
            - c_m1: c-rotations for movers attached to connection 1
            - c_m2: c-rotations for movers attached to connection 2
        """
        # mover 0
        c_m0 = -self.magbot_params.s_m0 * (a_d + b_d) + c_d
        # mover 1
        c_m1 = self.magbot_params.s_m1 * (a_d - b_d) + c_d
        # mover 2
        c_m2 = self.magbot_params.s_m2 * f_d + a_d + c_d

        return c_m0, c_m1, c_m2

    def gripperSetPose2MoverSetPose(
        self, gripper_pose_d: np.ndarray, z_m: np.ndarray | float, single_track: np.ndarray | bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the desired mover pose from a desired gripper pose. This method is vectorized and can therefore calculate desired
        mover poses for multiple MagBots.

        :param gripper_pose_d: the desired gripper poses (rotation specified as Euler angles (xyz) in rad). Shape: (num_samples,7)
            The 7th entry is the desired distance between the two finger pads in m (clipped to the reachable finger distance range).
        :param z_m: the desired mover z-pos (corresponds to the distance between the tile surface and the bottom of a mover).
            Can be a numpy array of shape (num_samples,) or a single float value (similar position for all movers).
        :param single_track: whether to calculate the mover set positions in single-track mode, defaults to False. Either a numpy
            array containing an individual bool value for each MagBot (shape: (num_magbots,)) or a single bool value.
        :return: the desired mover poses. Shape: (num_samples,3,6)
            [:,0,:]: poses for the mover attached to connection 0; [:,1,:]: poses for the mover attached to connection 1;
            [:,2,:]: poses for the mover attached to connection 2
        """
        num_samples = gripper_pose_d.shape[0]
        if isinstance(z_m, float):
            z_m = np.array([z_m] * num_samples)
        if isinstance(single_track, bool):
            single_track = np.array([single_track] * num_samples)

        mover_pose_euler_d = np.zeros((num_samples, 3, 6))
        mover_pose_euler_d[:, 0, 2] = z_m
        mover_pose_euler_d[:, 1, 2] = z_m
        mover_pose_euler_d[:, 2, 2] = z_m

        z_d = np.maximum(gripper_pose_d[:, 2], self.magbot_params.z_d_min)
        a_d = np.minimum(np.maximum(gripper_pose_d[:, 3], np.deg2rad(-180.0)), np.deg2rad(0.0))
        # the 7th gripper pose entry specifies the desired distance between the two finger pads (in m);
        # convert it to the corresponding finger (jaw) angle used by the finger actuator
        f_d = np.minimum(np.maximum(gripper_pose_d[:, 6], 0.0), self.magbot_params.f_d_max)
        f_d = self.finger_distance_to_angle(f_d)

        # mover distances
        d_m01, d_m2p = self._get_desired_mover_distances(z_d=z_d, a_d=a_d, z_m=z_m, single_track=single_track)
        # desired mover xy-positions
        x_m0, y_m0, x_m1, y_m1, x_m2, y_m2 = self._get_desired_mover_xy_positions(
            x_d=gripper_pose_d[:, 0], y_d=gripper_pose_d[:, 1], a_d=a_d, c_d=gripper_pose_d[:, 5], d_m01=d_m01, d_m2p=d_m2p
        )
        mover_pose_euler_d[:, 0, 0] = x_m0
        mover_pose_euler_d[:, 1, 0] = x_m1
        mover_pose_euler_d[:, 2, 0] = x_m2

        mover_pose_euler_d[:, 0, 1] = y_m0
        mover_pose_euler_d[:, 1, 1] = y_m1
        mover_pose_euler_d[:, 2, 1] = y_m2

        # desired mover gamma rotations
        c_m0, c_m1, c_m2 = self._get_desired_mover_c_rotations(a_d=a_d, b_d=gripper_pose_d[:, 4], c_d=gripper_pose_d[:, 5], f_d=f_d)
        mover_pose_euler_d[:, 0, 5] = c_m0
        mover_pose_euler_d[:, 1, 5] = c_m1
        mover_pose_euler_d[:, 2, 5] = c_m2

        return mover_pose_euler_d

    def get_gripper_pose(self, model: MjModel, data: MjData, indices_magbots: np.ndarray) -> np.ndarray:
        """Get the current pose of a MagBot's gripper (measured at the center between the two gripper fingers), including the current
        distance between the two fingers. This method is vectorized and can therefore calculate the current pose of multiple MagBots.

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        :param indices_magbots: the indices of the MagBots for which to return the pose. Shape: (num_indices,)
        :return: the current pose of the grippers (rotation represented as quaternion). Shape: (num_indices,8)
            (representation: (x_p,y_p,z_p,w_o,x_o,y_o,z_o,finger_distance)). The last entry is the current distance between the two
            finger pads in m.
        """
        if not hasattr(self, 'gripper_pose_site_ids'):
            self._update_cached_gripper_site_ids(model)
        if not hasattr(self, 'gripper_pad_1_geom_ids'):
            self._update_cached_gripper_finger_geom_ids(model)

        gripper_pose = np.zeros((indices_magbots.shape[0], 8))
        gripper_pose[:, :3] = data.site_xpos.reshape((-1, 3))[self.gripper_pose_site_ids[indices_magbots], :]

        site_xmats = data.site_xmat.reshape((-1, 3, 3))[self.gripper_pose_site_ids[indices_magbots], :, :]
        gripper_pose[:, 3:7] = rotations_utils.vec_mat2quat(site_xmats)

        # current distance between the two finger pads
        for idx, idx_magbot in enumerate(indices_magbots):
            gripper_pose[idx, 7] = mujoco.mj_geomDistance(
                model, data, self.gripper_pad_1_geom_ids[idx_magbot], self.gripper_pad_2_geom_ids[idx_magbot], 1.0, None
            )

        return gripper_pose
