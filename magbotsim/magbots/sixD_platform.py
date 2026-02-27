import numpy as np
import mujoco
from mujoco import MjModel, MjData
from gymnasium import logger
from magbotsim.utils import rotations_utils, mujoco_utils


class SixDPlatformMagBotsAPM4330:
    """Base class for 6D-Platform MagBots. This class can handle multiple 6D-Platform MagBots, allowing calculations to be vectorized.

    :param num_magbots: number of 6D-Platform MagBots
    :param indices_mover_a: a numpy array of shape (num_magbots,) containing all indices of movers controlling the alpha-rotation (around
        x-axis) of a platform
    :param indices_mover_b: a numpy array of shape (num_magbots,) containing all indices of movers controlling the beta-rotation (around
        y-axis) of a platform
    """

    def __init__(self, num_magbots: int, indices_mover_a: np.ndarray, indices_mover_b: np.ndarray) -> None:
        self.num_magbots = num_magbots
        self.magbots = []

        # minimum and maximum mover distance
        self.min_dist_movers_m = 0.258  # in m
        self.max_dist_movers_m = 0.455  # in m

        # gear ratios
        self.gear_ratio_mover2platform_a = 0.119
        self.gear_ratio_mover2platform_b = 0.131
        self.gear_ratio_platform2mover_a = 1 / self.gear_ratio_mover2platform_a
        self.gear_ratio_platform2mover_b = 1 / self.gear_ratio_mover2platform_b
        self.KP = 100.0

        # max rot
        self.max_platform_rot_ab = np.deg2rad(14.0)  # rad

        # mover indices
        self.mover_indices = np.concatenate((indices_mover_a[:, None], indices_mover_b[:, None]), axis=1)
        self.indices_mover_a = indices_mover_a
        self.indices_mover_b = indices_mover_b

        # platform A,B rot
        self.platform_joint_names = [f'ball_socket_joint_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]
        # platform qpos
        self.platform_pose_site_names = [f'platform_pose_site_{idx_magbot}' for idx_magbot in range(0, self.num_magbots)]

    def _update_cached_platform_site_ids(self, model: MjModel) -> None:
        """Update platform site ids.

        :param model: mjModel of the MuJoCo environment
        """
        self.platform_pose_site_ids = np.zeros((self.num_magbots,), dtype=np.int32)
        for idx_magbot in range(0, self.num_magbots):
            self.platform_pose_site_ids[idx_magbot] = model.site(self.platform_pose_site_names[idx_magbot]).id

    def _update_cached_platform_qpos_indices(self, model: MjModel) -> None:
        """Update platform qpos indices.

        :param model: mjModel of the MuJoCo environment
        """
        platform_joint_adrs = np.zeros((len(self.platform_joint_names),), dtype=np.int32)
        for idx_joint, joint_name in enumerate(self.platform_joint_names):
            joint_qpos_adr, _, qpos_ndim, qvel_qacc_ndim = mujoco_utils.get_joint_addrs_and_ndims(model, joint_name)
            assert qpos_ndim == 4 and qvel_qacc_ndim == 3, 'Platform joint ndims are not as expected'
            platform_joint_adrs[idx_joint] = joint_qpos_adr

        self.platform_qpos_indices = platform_joint_adrs[:, np.newaxis] + np.arange(qpos_ndim)

    def _update_cached_platform_ab_actuator_ids(self, model: MjModel) -> None:
        """Update actuator ids.

        :param model: mjModel of the MuJoCo environment
        """
        self.platform_ab_actuator_ids = np.zeros((self.num_magbots, 2), dtype=np.int32)
        for idx_magbot in range(0, self.num_magbots):
            for idx_a, actuator_name in enumerate(self.platform_ab_actuator_names[idx_magbot]):
                self.platform_ab_actuator_ids[idx_magbot, idx_a] = model.actuator(actuator_name).id

    def update_cached_mujoco_data(self, model: MjModel) -> None:
        """Update all cached information about MuJoCo objects, such as names and IDs.

        :param model: mjModel of the MuJoCo environment
        """
        # actuator IDs
        self._update_cached_platform_ab_actuator_ids(model)
        # platform A,B rot
        self._update_cached_platform_qpos_indices(model)
        # platform site IDs
        self._update_cached_platform_site_ids(model)
        # MagBot mass
        self.magbot_masses = np.zeros((self.num_magbots,))
        for idx_magbot in range(0, self.num_magbots):
            body_name = f'6D_platform_magbot_{idx_magbot}'
            self.magbot_masses[idx_magbot] = model.body(body_name).subtreemass[0]

    def _calculate_corrective_torque_platform_rot(
        self, mover_quat: np.ndarray, platform_joint_quat: np.ndarray, platform_c_rot: np.ndarray, is_a_rot: bool
    ) -> np.ndarray:
        """Calculate the corrective torque that needs to be applied to the platform, so that it rotates together with a mover. This method
        is vectorized and can therefore calculate corrective torques for multiple MagBots.

        :param mover_quat: a numpy array of shape (num_samples, 4) containing the rotations of the movers
            (quaternion representation (w,x,y,z))
        :param platform_joint_quat: a numpy array of shape (num_samples, 4) containing the current rotation of the
            (quaternion representation (w,x,y,z)).
        :param platform_c_rot: the current c rotation of the platform specified as a numpy array of shape (num_samples,)
        :param is_a_rot: whether to calculate the corrective torque for the platform's a (rot around x-axis) or b (rot around y-axis)
            rotation. This parameter influences the gear ratio.
        :return: a numpy array of shape (num_samples,) containing the corrective torques
        """
        assert mover_quat.shape[1] == 4
        assert platform_joint_quat.shape[1] == 4

        mover_euler = rotations_utils.vec_quat2euler(mover_quat)
        platform_joint_euler = rotations_utils.vec_quat2euler(platform_joint_quat)

        if is_a_rot:
            gear_ratio = self.gear_ratio_mover2platform_a
            idx_platform_euler = 0
        else:
            # b rot
            gear_ratio = self.gear_ratio_mover2platform_b
            idx_platform_euler = 1

        rot_error = platform_joint_euler[:, idx_platform_euler] - np.maximum(
            np.minimum(gear_ratio * (mover_euler[:, 2] - platform_c_rot), self.max_platform_rot_ab), -self.max_platform_rot_ab
        )
        torque = -self.KP * rot_error
        return torque

    def control_platform_ab_rot(
        self, model: MjModel, data: MjData, mover_a_quats: np.ndarray, mover_b_quats: np.ndarray, indices_magbots: np.ndarray | None = None
    ) -> None:
        """Couple the rotations around the z-axes of the movers with the platform's a and b rotations. This method is vectorized
        and can therefore couple the rotations for multiple MagBots.

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        :param mover_a_quats: current rotations of the movers (represented as quaternions (w,x,y,z)) that control the platform's
            alpha-rotation. Shape: (num_indices, 4)
        :param mover_b_quats: current rotations of the movers (represented as quaternions (w,x,y,z)) that control the platform's
            beta-rotation. Shape: (num_indices, 4)
        :param indices_magbots: the indices of the MagBots to control, defaults to None. If None, all MagBots are controlled.
            Shape: (num_indices,)
        """
        if indices_magbots is None:
            indices_magbots = np.arange(0, self.num_magbots, 1)
        # get current platform pose
        if not hasattr(self, 'platform_qpos_indices'):
            self._update_cached_platform_qpos_indices(model)

        platform_joint_quats = data.qpos[self.platform_qpos_indices[indices_magbots, :]]
        platform_c_rot = rotations_utils.vec_quat2euler(self.get_platform_pose(model, data, indices_magbots=indices_magbots)[:, 3:])[:, 2]

        # calculate corrective torques
        torques_a = self._calculate_corrective_torque_platform_rot(
            mover_quat=mover_a_quats, platform_joint_quat=platform_joint_quats, platform_c_rot=platform_c_rot, is_a_rot=True
        )
        torques_b = self._calculate_corrective_torque_platform_rot(
            mover_quat=mover_b_quats, platform_joint_quat=platform_joint_quats, platform_c_rot=platform_c_rot, is_a_rot=False
        )

        # apply corrective torques
        if not hasattr(self, 'platform_ab_actuator_ids'):
            self._update_cached_platform_ab_actuator_ids(model)

        data.ctrl[self.platform_ab_actuator_ids[indices_magbots, 0]] = torques_a
        data.ctrl[self.platform_ab_actuator_ids[indices_magbots, 1]] = torques_b

    def generate_magbot_xml_strings(self, initial_pos_xyz_mover_b: np.ndarray, custom_model_xml_strings: dict | None) -> dict:
        """Generate the MuJoCo XML strings for all 6D-Platform MagBots and add them to the ``custom_model_xml_strings``-dict.

        :param initial_pos_xyz_mover_b: the initial x,y,z-positions for all movers that control the b rotation of the platforms.
            Shape: (num_magbots,3)
        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this method
        :return: the modified ``custom_model_xml_strings``-dict
        """
        assert initial_pos_xyz_mover_b.shape == (self.num_magbots, 3)

        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}

        for idx_magbot in range(0, self.num_magbots):
            # assets
            if idx_magbot == 0:
                platform_assets_list = [
                    '\n\t\t<!-- assets 6D-Platform MagBot -->',
                    '\t\t<mesh name="mover_to_mount_a" file="./magbots/6D_platform_XPlanar_APM4330/mover_to_mount_b.STL" scale="1 1 1" />',
                    '\t\t<mesh name="mover_to_mount_b" file="./magbots/6D_platform_XPlanar_APM4330/mover_to_mount_a.STL" scale="1 1 1" />',
                    '\t\t<mesh name="rail_mount_mesh_a" file="./magbots/6D_platform_XPlanar_APM4330/rail_mount_a.STL" scale="1 1 1" />',
                    '\t\t<mesh name="rail_mount_mesh_b" file="./magbots/6D_platform_XPlanar_APM4330/rail_mount_b.STL" scale="1 1 1" />',
                    '\t\t<mesh name="rotation_plate_mesh" file="./magbots/6D_platform_XPlanar_APM4330/rotation_plate.STL" scale="1 1 1" />',
                    '\t\t<mesh name="solid_connection" file="./magbots/6D_platform_XPlanar_APM4330/solid_connection_v2.STL" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="connection_mid_1_mesh" file="./magbots/6D_platform_XPlanar_APM4330/mid_connection_v2.STL" '
                    'scale="1 1 1"/>',
                    '\t\t<mesh name="platform_bottom_mesh" file="./magbots/6D_platform_XPlanar_APM4330/platform_bottom.STL" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="platform_top_mesh" file="./magbots/6D_platform_XPlanar_APM4330/platform_top.STL" scale="1 1 1" />',
                    '\t\t<mesh name="ball_stud_mesh" file="./magbots/6D_platform_XPlanar_APM4330/ball_stud.STL" scale="1 1 1" />',
                    '\t\t<mesh name="ball_socket_mesh" file="./magbots/6D_platform_XPlanar_APM4330/ball_socket.STL" scale="1 1 1" />',
                    '\t\t<mesh name="flange_coupling_mesh" file="./magbots/6D_platform_XPlanar_APM4330/flange_coupling_M6.STL" '
                    'scale="1 1 1"/>',
                    '\t\t<mesh name="ball_stud_m4_mesh" file="./magbots/6D_platform_XPlanar_APM4330/ball_stud_M4.STL" scale="1 1 1" />',
                    '\t\t<mesh name="linear_coupling_mesh" file="./magbots/6D_platform_XPlanar_APM4330/linear_coupling_M4.STL" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="toothed_rack_mesh" file="./magbots/6D_platform_XPlanar_APM4330/toothed_rack.STL" scale="1 1 1" />',
                    '\t\t<mesh name="bevel_gear_mesh" file="./magbots/6D_platform_XPlanar_APM4330/bevel_gear.STL" scale="1 1 1" />',
                    '\t\t<mesh name="spur_gear_mesh" file="./magbots/6D_platform_XPlanar_APM4330/spur_gear.STL" scale="1 1 1" />',
                    '\t\t<mesh name="gliding_bearing_mesh" file="./magbots/6D_platform_XPlanar_APM4330/gliding_bearing_toothed_rack.STL" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="GT2_Pulley_mesh" file="./magbots/6D_platform_XPlanar_APM4330/GT2_Pulley_5mm_Bore.STL" scale="1 1 1"/>',
                    '\t\t<mesh name="shaft_5mm_short_mesh" file="./magbots/6D_platform_XPlanar_APM4330/shaft_5mm_short.STL" '
                    'scale="1 1 1"/>',
                    '\t\t<mesh name="shaft_5mm_long_mesh" file="./magbots/6D_platform_XPlanar_APM4330/shaft_5mm_long.STL" scale="1 1 1" />',
                    '\t\t<mesh name="toothed_belt_short_mesh" file="./magbots/6D_platform_XPlanar_APM4330/toothed_belt_short.STL" '
                    'scale="1 1 1" />',
                    '\t\t<mesh name="toothed_belt_long_mesh" file="./magbots/6D_platform_XPlanar_APM4330/toothed_belt_long.STL" '
                    'scale="1 1 1" />',
                ]
                platform_assets_xml_str = '\n'.join(platform_assets_list)
                custom_assets_xml_str = custom_model_xml_strings.get('custom_assets_xml_str', None)
                if custom_assets_xml_str is not None:
                    custom_assets_xml_str += platform_assets_xml_str
                else:
                    custom_assets_xml_str = platform_assets_xml_str
                custom_model_xml_strings['custom_assets_xml_str'] = custom_assets_xml_str

            # mover body XML strings
            mover_body_xml_strs = custom_model_xml_strings.get('custom_mover_body_xml_str_list')
            assert isinstance(mover_body_xml_strs, list), 'custom_mover_body_xml_str_list must be a list'
            # mover a
            mover_body_a_list = [
                f'\n\t\t\t<body name="m_{self.indices_mover_a[idx_magbot]}_rail" pos="0 0 0.006" gravcomp="1" euler="0 0 0">',
                f'\t\t\t\t<geom name="m_{self.indices_mover_a[idx_magbot]}_to_mount_geom" type="mesh" mesh="mover_to_mount_a" mass="0.065" '
                'pos="0 0 0" material="green"/>',
                f'\t\t\t\t<site name="m_{self.indices_mover_a[idx_magbot]}_rail_site" pos="0 0 0" euler="0 0 -1.5708" size="0.001" '
                'type="sphere"/>',
                '\t\t\t</body>',
            ]
            mover_body_a_xml_str = '\n'.join(mover_body_a_list)
            mover_body_xml_strs[self.indices_mover_a[idx_magbot]] = mover_body_a_xml_str
            # mover b
            mover_body_b_list = [
                f'\n\t\t\t<body name="m_{self.indices_mover_b[idx_magbot]}_rail" pos="0 0 0.006" gravcomp="1" euler="0 0 1.5708">',
                f'\t\t\t\t<geom name="m_{self.indices_mover_b[idx_magbot]}_to_mount_geom" type="mesh" mesh="mover_to_mount_b" mass="0.065" '
                'pos="0 0 0" material="green"/>',
                f'\t\t\t\t<site name="m_{self.indices_mover_b[idx_magbot]}_rail_site" pos="0 0 0" size="0.001" type="sphere"/>',
                '\t\t\t</body>',
            ]
            mover_body_b_xml_str = '\n'.join(mover_body_b_list)
            mover_body_xml_strs[self.indices_mover_b[idx_magbot]] = mover_body_b_xml_str
            custom_model_xml_strings['custom_mover_body_xml_str_list'] = mover_body_xml_strs

            # platform magbot
            platform_magbot_list = [
                f'\n\t\t<!-- 6D-Platform MagBot {idx_magbot} -->',
                f'\t\t<body name="6D_platform_magbot_{idx_magbot}" pos="{initial_pos_xyz_mover_b[idx_magbot, 0]} '
                f'{initial_pos_xyz_mover_b[idx_magbot, 1]} {initial_pos_xyz_mover_b[idx_magbot, 2] + 0.012}" gravcomp="1" '
                'euler="0 0 3.1416">',
                '\t\t\t<inertial mass="0.001" diaginertia="1e-6 1e-6 1e-6" pos="0 0 0"/>',
                f'\t\t\t<joint name="platform_x_{idx_magbot}" type="slide" axis="1 0 0"/>',
                f'\t\t\t<joint name="platform_y_{idx_magbot}" type="slide" axis="0 1 0"/>',
                f'\t\t\t<joint name="platform_yaw_{idx_magbot}" type="hinge" axis="0 0 1"/>',
                f'\t\t\t<body name="rail_mount_0_{idx_magbot}" pos="0.0 0.0 0.0" gravcomp="1" euler="0 0 -1.5708">',
                f'\t\t\t\t<joint name="rail_mount_0_joint_{idx_magbot}" type="slide" axis="1 0 0" damping="0.1"/>',
                f'\t\t\t\t<joint name="rail_mount_0_joint_yaw_{idx_magbot}" type="hinge" axis="0 0 1"/>',
                f'\t\t\t\t<geom name="rail_mount_geom_0_{idx_magbot}" type="mesh" mesh="rail_mount_mesh_a" mass="0.5" pos="0 0 0" '
                'material="black"/>',
                f'\t\t\t\t<site name="rail_mount_0_site_{idx_magbot}" pos="0 0 0" size="0.001" type="sphere"/>',
                f'\t\t\t\t<body name="bevel_gear_A_0_{idx_magbot}" pos="0.0 0.0 0.038" gravcomp="1" euler="0 0 0">',
                f'\t\t\t\t\t<geom name="bevel_gear_A_0_geom_{idx_magbot}" type="mesh" mesh="bevel_gear_mesh" mass="0.01" pos="0 0 0" '
                'material="off_white"/>',
                '\t\t\t\t</body>',
                f'\t\t\t\t<body name="rotation_plate_0_{idx_magbot}" pos="0.0 0.0 0.03" gravcomp="1" euler="0 0 1.5708">',
                f'\t\t\t\t\t<joint name="rotation_plate_0_joint_{idx_magbot}" type="hinge" axis="0 0 1" pos="0 0 0" damping="0" />',
                f'\t\t\t\t\t<geom name="rotation_plate_geom_0_{idx_magbot}" type="mesh" mesh="rotation_plate_mesh" mass="0.01" pos="0 0 0" '
                'material="black"/>',
                f'\t\t\t\t\t<body name="shaft_B_0_{idx_magbot}" pos="0.0 0.0475 0.0269" gravcomp="1" euler="1.5708 0 0">',
                f'\t\t\t\t\t\t<geom name="shaft_B_0_geom_{idx_magbot}" type="mesh" mesh="shaft_5mm_short_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t<body name="bevel_gear_A_1_{idx_magbot}" pos="0.0 0.0 0.0659" gravcomp="1" euler="3.1416 0 0">',
                f'\t\t\t\t\t\t\t<geom name="bevel_gear_A_1_geom_{idx_magbot}" type="mesh" mesh="bevel_gear_mesh" mass="0.01" pos="0 0 0" '
                'material="off_white"/>',
                '\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t<body name="gt2_pulley_B_0_{idx_magbot}" pos="0.0 0.0 0.039" gravcomp="1" euler="3.1416 0 0">',
                f'\t\t\t\t\t\t\t<geom name="gt2_pulley_B_0_geom_{idx_magbot}" type="mesh" mesh="GT2_Pulley_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t\t<body name="toothed_belt_long_B_{idx_magbot}" pos="0.0 0.0 0.011" gravcomp="1" '
                'euler="1.5708 0.7854 1.5708">',
                f'\t\t\t\t\t\t\t\t<joint name="toothed_belt_long_B_joint_{idx_magbot}" type="hinge" axis="1 0 0" pos="0 0 0.0" '
                'damping="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="toothed_belt_long_B_geom_{idx_magbot}" type="mesh" mesh="toothed_belt_long_mesh" '
                'mass="0.01" pos="0 0 0" material="black"/>',
                '\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t</body>',
                '\t\t\t\t\t</body>',
                f'\t\t\t\t\t<body name="solid_connection_0_{idx_magbot}" pos="0.075154 0.0 0.082154" gravcomp="1" euler="0 -0.7854 0">',
                f'\t\t\t\t\t\t<joint name="solid_connection_0_joint_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.09299 0 0" damping="0" '
                'range="-0.3 0.6109"/>',
                f'\t\t\t\t\t\t<geom name="solid_connection_0_geom_{idx_magbot}" type="mesh" mesh="solid_connection" mass="0.01" pos="0 0 0"'
                ' material="green"/>',
                '\t\t\t\t\t</body>',
                f'\t\t\t\t\t<body name="second_connection_0_{idx_magbot}" pos="0.105154 0.0 0.082154" gravcomp="1" euler="0 -0.7854 0">',
                f'\t\t\t\t\t\t<joint name="second_connection_0_joint_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.09299 0 0" damping="0"'
                ' range="-0.3 0.6109"/>',
                f'\t\t\t\t\t\t<geom name="second_connection_0_geom_{idx_magbot}" type="mesh" mesh="connection_mid_1_mesh" mass="0.01" '
                'pos="0 0 0" material="green"/>',
                f'\t\t\t\t\t\t<body name="platform_bottom_{idx_magbot}" pos="0.10699 0 -0.02903" gravcomp="1" euler="0 0.7854 0">',
                f'\t\t\t\t\t\t\t<joint name="connection_0_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.041 0.05123 0" damping="0"/>',
                f'\t\t\t\t\t\t\t<geom name="platform_bottom_geom_{idx_magbot}" type="mesh" mesh="platform_bottom_mesh" mass="0.01" '
                'pos="0 0 0" material="black" />',
                f'\t\t\t\t\t\t\t<site name="pb_hinge_p_{idx_magbot}" pos="0.071 0.05123 0.0" size="0.001" type="sphere"/>',
                f'\t\t\t\t\t\t\t<site name="pb_hinge_m_{idx_magbot}" pos="0.071 -0.05123 0.0" size="0.001" type="sphere"/>',
                f'\t\t\t\t\t\t\t<body name="shaft_B_1_{idx_magbot}" pos="-0.08988 0.0475 0.0" gravcomp="1" euler="1.5708 0 0">',
                f'\t\t\t\t\t\t\t\t<geom name="shaft_B_1_geom_{idx_magbot}" type="mesh" mesh="shaft_5mm_short_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t\t\t<body name="gt2_pulley_B_1_{idx_magbot}" pos="0.0 0.0 0.039" gravcomp="1" euler="3.1416 0 0">',
                f'\t\t\t\t\t\t\t\t\t<geom name="gt2_pulley_B_1_geom_{idx_magbot}" type="mesh" mesh="GT2_Pulley_mesh" mass="0.01" '
                'pos="0 0 0" material="gray"/>',
                '\t\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t\t<body name="spur_gear_B_{idx_magbot}" pos="0 0 0.0475" gravcomp="1" euler="0 -1.5708 0">',
                f'\t\t\t\t\t\t\t\t\t<geom name="spur_gear_B_geom_{idx_magbot}" type="mesh" mesh="spur_gear_mesh" mass="0.01" pos="0 0 0" '
                'material="off_white"/>',
                '\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t<body name="shaft_A_1_{idx_magbot}" pos="0.08988 0.0475 0.0" gravcomp="1" euler="1.5708 0 0">',
                f'\t\t\t\t\t\t\t\t<geom name="shaft_A_1_geom_{idx_magbot}" type="mesh" mesh="shaft_5mm_short_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t\t\t<body name="gt2_pulley_A_1_{idx_magbot}" pos="0.0 0.0 0.056" gravcomp="1" euler="0 0 0">',
                f'\t\t\t\t\t\t\t\t\t<geom name="gt2_pulley_A_1_geom_{idx_magbot}" type="mesh" mesh="GT2_Pulley_mesh" mass="0.01" '
                'pos="0 0 0" material="gray"/>',
                '\t\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t\t<body name="gt2_pulley_A_2_{idx_magbot}" pos="0.0 0.0 0.040" gravcomp="1" euler="3.1416 0 0">',
                f'\t\t\t\t\t\t\t\t\t<geom name="gt2_pulley_A_2_geom_{idx_magbot}" type="mesh" mesh="GT2_Pulley_mesh" mass="0.01" '
                'pos="0 0 0" material="gray"/>',
                f'\t\t\t\t\t\t\t\t\t<body name="toothed_belt_short_A_{idx_magbot}" pos="0.0 0.0 0.011" gravcomp="1" euler="0 -1.5708 0">',
                f'\t\t\t\t\t\t\t\t\t\t<geom name="toothed_belt_short_A_geom_{idx_magbot}" type="mesh" mesh="toothed_belt_short_mesh" '
                'mass="0.01" pos="0 0 0" material="black"/>',
                '\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t<body name="shaft_A_2_{idx_magbot}" pos="0.01508 0.105 0.0" gravcomp="1" euler="1.5708 0 0">',
                f'\t\t\t\t\t\t\t\t<geom name="shaft_A_2_geom_{idx_magbot}" type="mesh" mesh="shaft_5mm_long_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t\t\t<body name="gt2_pulley_A_3_{idx_magbot}" pos="0.0 0.0 0.0975" gravcomp="1" euler="3.1416 0 0">',
                f'\t\t\t\t\t\t\t\t\t<geom name="gt2_pulley_A_3_geom_{idx_magbot}" type="mesh" mesh="GT2_Pulley_mesh" mass="0.01" '
                'pos="0 0 0" material="gray"/>',
                '\t\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t\t<body name="spur_gear_A_{idx_magbot}" pos="0 0 0.027" gravcomp="1" euler="0 -1.5708 0">',
                f'\t\t\t\t\t\t\t\t\t<geom name="spur_gear_A_geom_{idx_magbot}" type="mesh" mesh="spur_gear_mesh" mass="0.01" pos="0 0 0" '
                'material="off_white"/>',
                '\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t<body name="gliding_bearing_rack_A_{idx_magbot}" pos="0.002 0.077 -0.0333" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t<geom name="gliding_bearing_rack_A_geom_{idx_magbot}" type="mesh" mesh="gliding_bearing_mesh" mass="0.01"'
                ' pos="0 0 0" material="green"/>',
                '\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t<body name="toothed_rack_A_rot_{idx_magbot}" pos="0.0012 0.0772 0.0433" gravcomp="1" euler="0 0 -1.5708">',
                f'\t\t\t\t\t\t\t\t<joint name="toothed_rack_A_rot_joint_{idx_magbot}" type="slide" axis="0 0 1" pos="0 0 0" damping="0.1" '
                'limited="true" range="-0.022 0.02818"/>',
                f'\t\t\t\t\t\t\t\t<geom name="toothed_rack_A_rot_geom_{idx_magbot}" type="mesh" mesh="toothed_rack_mesh" mass="0.01" '
                'pos="0 0 0" material="off_white"/>',
                f'\t\t\t\t\t\t\t\t<body name="ball_stud_A_rot_{idx_magbot}" pos="0.0 0.0 -0.006" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t<geom name="ball_stud_A_rot_geom_{idx_magbot}" type="mesh" mesh="ball_stud_m4_mesh" mass="0.01" '
                'pos="0 0 0" material="gray"/>',
                f'\t\t\t\t\t\t\t\t\t<site name="A_ball_site_{idx_magbot}" pos="0 0 0.014" size="0.001" type="sphere"/>',
                '\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t<body name="gliding_bearing_rack_B_{idx_magbot}" pos="-0.075 0.001 -0.0333" gravcomp="1" '
                'euler="0 0 3.1416">',
                f'\t\t\t\t\t\t\t\t<geom name="gliding_bearing_rack_B_geom_{idx_magbot}" type="mesh" mesh="gliding_bearing_mesh" mass="0.01"'
                ' pos="0 0 0" material="green"/>',
                '\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t<body name="toothed_rack_B_rot_{idx_magbot}" pos="-0.0745 0.0008 0.0433" gravcomp="1" euler="0 0 1.5708">',
                f'\t\t\t\t\t\t\t\t<joint name="toothed_rack_B_rot_joint_{idx_magbot}" type="slide" axis="0 0 1" pos="0 0 0" damping="0.1" '
                'limited="true" range="-0.022 0.02818"/>',
                f'\t\t\t\t\t\t\t\t<geom name="toothed_rack_B_rot_geom_{idx_magbot}" type="mesh" mesh="toothed_rack_mesh" mass="0.01" '
                'pos="0 0 0" material="off_white"/>',
                f'\t\t\t\t\t\t\t\t<body name="ball_stud_B_rot_{idx_magbot}" pos="0.0 0.0 -0.006" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t<geom name="ball_stud_B_rot_geom_{idx_magbot}" type="mesh" mesh="ball_stud_m4_mesh" mass="0.01" '
                'pos="0 0 0" material="gray"/>',
                f'\t\t\t\t\t\t\t\t\t<site name="B_ball_site_{idx_magbot}" pos="0 0 0.014" size="0.001" type="sphere"/>',
                '\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t<body name="flange_coupling_{idx_magbot}" pos="0.0 0.0 0.017" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t<geom name="flange_coupling_geom_{idx_magbot}" type="mesh" mesh="flange_coupling_mesh" mass="0.01" '
                'pos="0 0 0" material="gray"/>',
                f'\t\t\t\t\t\t\t\t<body name="ball_stud_{idx_magbot}" pos="0.0 0.0 0.0077" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t<geom name="ball_stud_geom_{idx_magbot}" type="mesh" mesh="ball_stud_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t\t\t\t<body name="ball_socket_{idx_magbot}" pos="0.0 0.0 0.0215" gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t<joint name="ball_socket_joint_{idx_magbot}" type="ball" pos="0 0 0" damping="0.0" '
                'solimplimit="0.95 1 0.001" solreflimit="0.001 5"/>',
                f'\t\t\t\t\t\t\t\t\t\t<geom name="ball_socket_geom_{idx_magbot}" type="mesh" mesh="ball_socket_mesh" mass="0.01" '
                'pos="0 0 0" material="black"/>',
                f'\t\t\t\t\t\t\t\t\t\t<body name="platform_top_{idx_magbot}" pos="0.0 0.0 0.0075" gravcomp="1" euler="0 0 1.5708">',
                f'\t\t\t\t\t\t\t\t\t\t\t<geom name="platform_top_geom_{idx_magbot}" type="mesh" mesh="platform_top_mesh" mass="0.01" '
                'pos="0 0 0" material="black"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<site name="platform_pose_site_{idx_magbot}" pos="0 0 0.004" euler="0 0 1.5708" size="0.001" '
                'type="sphere" rgba="1 0 0 0"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t<body name="guide_A_{idx_magbot}" pos="0 0 0">',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<joint name="linear_coupling_A_slide_{idx_magbot}" type="slide" axis="1 0 0" damping="0.1" '
                'range="-0.0045 0.0045"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<body name="linear_coupling_A_{idx_magbot}" pos="0.077 -0.001 -0.004" euler="0 0 1.5708" '
                'gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t\t\t\t<geom name="linear_coupling_A_geom_{idx_magbot}" type="mesh" mesh="linear_coupling_mesh" '
                'mass="0.01" material="black"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t\t\t<site name="A_socket_site_{idx_magbot}" pos="0 0 0.002" size="0.001" type="sphere"/>',
                '\t\t\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t\t\t\t\t\t<body name="guide_B_{idx_magbot}" pos="0 0 0">',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<joint name="linear_coupling_B_slide_{idx_magbot}" type="slide" axis="0 1 0" damping="0.1" '
                'range="-0.0045 0.0045"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t\t<body name="linear_coupling_B_{idx_magbot}" pos="0.0008 0.0748 -0.004" euler="0 0 0" '
                'gravcomp="1">',
                f'\t\t\t\t\t\t\t\t\t\t\t\t\t<geom name="linear_coupling_B_geom_{idx_magbot}" type="mesh" mesh="linear_coupling_mesh" '
                'mass="0.01" material="black"/>',
                f'\t\t\t\t\t\t\t\t\t\t\t\t\t<site name="B_socket_site_{idx_magbot}" pos="0 0 0.002" size="0.001" type="sphere"/>',
                '\t\t\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t</body>',
                '\t\t\t\t\t</body>',
                '\t\t\t\t</body>',
                '\t\t\t</body>',
                f'\t\t\t<body name="rail_mount_1_{idx_magbot}" pos="0.402616 0.0 0.0" gravcomp="1" euler="0 0 3.1416">',
                f'\t\t\t\t<joint name="rail_mount_1_joint_{idx_magbot}" type="slide" axis="1 0 0" damping="0.1"/>',
                f'\t\t\t\t<joint name="rail_mount_1_joint_yaw_{idx_magbot}" type="hinge" axis="0 0 1"/>',
                f'\t\t\t\t<geom name="rail_mount_geom_1_{idx_magbot}" type="mesh" mesh="rail_mount_mesh_b" mass="0.5" pos="0 0 0" '
                'material="black"/>',
                f'\t\t\t\t<site name="rail_mount_1_site_{idx_magbot}" pos="0 0 0" size="0.001" type="sphere"/>',
                f'\t\t\t\t<body name="bevel_gear_B_0_{idx_magbot}" pos="0.0 0.0 0.038" gravcomp="1" euler="0 0 0">',
                f'\t\t\t\t\t<geom name="bevel_gear_B_0_geom_{idx_magbot}" type="mesh" mesh="bevel_gear_mesh" mass="0.01" pos="0 0 0" '
                'material="off_white"/>',
                '\t\t\t\t</body>',
                f'\t\t\t\t<body name="rotation_plate_1_{idx_magbot}" pos="0.0 0.0 0.03" gravcomp="1">',
                f'\t\t\t\t\t<joint name="rotation_plate_1_joint_{idx_magbot}" type="hinge" axis="0 0 1" pos="0 0 0" damping="0" />',
                f'\t\t\t\t\t<geom name="rotation_plate_geom_1_{idx_magbot}" type="mesh" mesh="rotation_plate_mesh" mass="0.01" pos="0 0 0" '
                'material="black"/>',
                f'\t\t\t\t\t<body name="shaft_A_0_{idx_magbot}" pos="0.0 0.0475 0.0269" gravcomp="1" euler="1.5708 0 0">',
                f'\t\t\t\t\t\t<geom name="shaft_A_0_geom_{idx_magbot}" type="mesh" mesh="shaft_5mm_short_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t<body name="bevel_gear_B_1_{idx_magbot}" pos="0.0 0.0 0.0659" gravcomp="1" euler="3.1416 0 0">',
                f'\t\t\t\t\t\t\t<geom name="bevel_gear_B_1_geom_{idx_magbot}" type="mesh" mesh="bevel_gear_mesh" mass="0.01" pos="0 0 0" '
                'material="off_white"/>',
                '\t\t\t\t\t\t</body>',
                f'\t\t\t\t\t\t<body name="gt2_pulley_A_0_{idx_magbot}" pos="0.0 0.0 0.039" gravcomp="1" euler="3.1416 0 0">',
                f'\t\t\t\t\t\t\t<geom name="gt2_pulley_A_0_geom_{idx_magbot}" type="mesh" mesh="GT2_Pulley_mesh" mass="0.01" pos="0 0 0" '
                'material="gray"/>',
                f'\t\t\t\t\t\t\t<body name="toothed_belt_long_A_{idx_magbot}" pos="0.0 0.0 0.011" gravcomp="1" '
                'euler="1.5708 0.7854 1.5708">',
                f'\t\t\t\t\t\t\t\t<joint name="toothed_belt_long_A_joint_{idx_magbot}" type="hinge" axis="1 0 0" pos="0 0 0.0" '
                'damping="0"/>',
                f'\t\t\t\t\t\t\t\t<geom name="toothed_belt_long_A_geom_{idx_magbot}" type="mesh" mesh="toothed_belt_long_mesh" mass="0.01" '
                'pos="0 0 0" material="black"/>',
                '\t\t\t\t\t\t\t</body>',
                '\t\t\t\t\t\t</body>',
                '\t\t\t\t\t</body>',
                f'\t\t\t\t\t<body name="solid_connection_1_{idx_magbot}" pos="0.075154 0.0 0.082154" gravcomp="1" euler="0 -0.7854 0">',
                f'\t\t\t\t\t\t<joint name="solid_connection_1_joint_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.09299 0 0" damping="0" '
                'range="-0.3 0.6109"/>',
                f'\t\t\t\t\t\t<geom name="solid_connection_geom_1_1_{idx_magbot}" type="mesh" mesh="solid_connection" mass="0.01" '
                'pos="0 0 0" material="green"/>',
                f'\t\t\t\t\t\t<site name="sc1_hinge_p_{idx_magbot}" pos="0.078 -0.05123 0" size="0.001" type="sphere"/>',
                f'\t\t\t\t\t\t<site name="sc1_hinge_m_{idx_magbot}" pos="0.078 0.05123 0" size="0.001" type="sphere"/>',
                '\t\t\t\t\t</body>',
                f'\t\t\t\t\t<body name="second_connection_1_{idx_magbot}" pos="0.105154 0.0 0.082154" gravcomp="1" euler="0 -0.7854 0">',
                f'\t\t\t\t\t\t<joint name="second_connection_1_joint_{idx_magbot}" type="hinge" axis="0 1 0" pos="-0.09299 0 0" damping="0"'
                ' range="-0.3 0.6109"/>',
                f'\t\t\t\t\t\t<geom name="second_connection_1_geom_{idx_magbot}" type="mesh" mesh="connection_mid_1_mesh" mass="0.01" '
                'pos="0 0 0" material="green"/>',
                '\t\t\t\t\t</body>',
                '\t\t\t\t</body>',
                '\t\t\t</body>',
                '\t\t</body>',
            ]
            platform_magbot_xml_str = '\n'.join(platform_magbot_list)
            custom_worldbody_xml_str = custom_model_xml_strings.get('custom_worldbody_xml_str', None)
            if custom_worldbody_xml_str is not None:
                custom_worldbody_xml_str += platform_magbot_xml_str
            else:
                custom_worldbody_xml_str = platform_magbot_xml_str
            custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

            # contacts
            platform_contact_list = [
                f'\n\t\t<!-- contacts 6D-Platform MagBot {idx_magbot}  -->',
                f'\t\t<exclude body1="platform_bottom_{idx_magbot}" body2="solid_connection_0_{idx_magbot}"/>',
                f'\t\t<exclude body1="platform_bottom_{idx_magbot}" body2="second_connection_0_{idx_magbot}"/>',
                f'\t\t<exclude body1="platform_bottom_{idx_magbot}" body2="solid_connection_1_{idx_magbot}"/>',
                f'\t\t<exclude body1="second_connection_1_{idx_magbot}" body2="platform_bottom_{idx_magbot}"/>',
                f'\t\t<exclude body1="rotation_plate_0_{idx_magbot}" body2="second_connection_0_{idx_magbot}"/>',
                f'\t\t<exclude body1="rail_mount_1_{idx_magbot}" body2="m_{self.indices_mover_a[idx_magbot]}_rail"/>',
                f'\t\t<exclude body1="rail_mount_0_{idx_magbot}" body2="m_{self.indices_mover_b[idx_magbot]}_rail"/>',
                f'\t\t<exclude body1="second_connection_0_{idx_magbot}" body2="toothed_rack_B_rot_{idx_magbot}"/>',
                f'\t\t<exclude body1="solid_connection_0_{idx_magbot}" body2="toothed_rack_B_rot_{idx_magbot}"/>',
                f'\t\t<exclude body1="linear_coupling_A_{idx_magbot}" body2="ball_stud_A_rot_{idx_magbot}"/>',
                f'\t\t<exclude body1="linear_coupling_B_{idx_magbot}" body2="ball_stud_B_rot_{idx_magbot}"/>',
                f'\t\t<exclude body1="gliding_bearing_rack_A_{idx_magbot}" body2="toothed_rack_A_rot_{idx_magbot}"/>',
                f'\t\t<exclude body1="gliding_bearing_rack_B_{idx_magbot}" body2="toothed_rack_B_rot_{idx_magbot}"/>',
                f'\t\t<exclude body1="second_connection_0_{idx_magbot}" body2="gliding_bearing_rack_B_{idx_magbot}"/>',
                f'\t\t<exclude body1="solid_connection_0_{idx_magbot}" body2="gliding_bearing_rack_B_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_long_B_{idx_magbot}" body2="gt2_pulley_B_1_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_long_B_{idx_magbot}" body2="shaft_B_1_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_long_B_{idx_magbot}" body2="platform_bottom_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_long_A_{idx_magbot}" body2="gt2_pulley_A_1_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_long_A_{idx_magbot}" body2="shaft_A_1_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_long_A_{idx_magbot}" body2="platform_bottom_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_short_A_{idx_magbot}" body2="gt2_pulley_A_3_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_short_A_{idx_magbot}" body2="shaft_A_2_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_short_A_{idx_magbot}" body2="platform_bottom_{idx_magbot}"/>',
                f'\t\t<exclude body1="toothed_belt_short_A_{idx_magbot}" body2="solid_connection_1_{idx_magbot}"/>',
                f'\t\t<exclude body1="spur_gear_B_{idx_magbot}" body2="solid_connection_0_{idx_magbot}"/>',
                f'\t\t<exclude body1="rail_mount_0_{idx_magbot}" body2="rotation_plate_0_{idx_magbot}"/>',
                f'\t\t<exclude body1="rail_mount_0_{idx_magbot}" body2="bevel_gear_A_0_{idx_magbot}"/>',
                f'\t\t<exclude body1="rail_mount_1_{idx_magbot}" body2="rotation_plate_1_{idx_magbot}"/>',
                f'\t\t<exclude body1="rail_mount_1_{idx_magbot}" body2="bevel_gear_B_0_{idx_magbot}"/>',
            ]
            platform_contact_xml_str = '\n'.join(platform_contact_list)
            custom_contact_xml_str = custom_model_xml_strings.get('custom_contact_xml_str', None)
            if custom_contact_xml_str is not None:
                custom_contact_xml_str += platform_contact_xml_str
            else:
                custom_contact_xml_str = platform_contact_xml_str
            custom_model_xml_strings['custom_contact_xml_str'] = custom_contact_xml_str

            # equality constraints
            equality_list = [
                '\n\t<equality>',
                f'\t\t<joint joint1="solid_connection_0_joint_{idx_magbot}" joint2="second_connection_0_joint_{idx_magbot}"/>',
                f'\t\t<joint joint1="solid_connection_0_joint_{idx_magbot}" joint2="solid_connection_1_joint_{idx_magbot}"/>',
                f'\t\t<joint joint1="second_connection_1_joint_{idx_magbot}" joint2="solid_connection_1_joint_{idx_magbot}"/>',
                f'\t\t<joint joint1="linear_coupling_A_slide_{idx_magbot}" joint2="toothed_rack_A_rot_joint_{idx_magbot}" '
                'polycoef="0 0.010"/>',
                f'\t\t<joint joint1="linear_coupling_B_slide_{idx_magbot}" joint2="toothed_rack_B_rot_joint_{idx_magbot}" '
                'polycoef="0 0.010"/>',
                f'\t\t<connect site1="A_socket_site_{idx_magbot}" site2="A_ball_site_{idx_magbot}"/>',
                f'\t\t<connect site1="B_socket_site_{idx_magbot}" site2="B_ball_site_{idx_magbot}"/>',
                f'\t\t<joint joint1="toothed_belt_long_B_joint_{idx_magbot}" joint2="solid_connection_0_joint_{idx_magbot}"/>',
                f'\t\t<joint joint1="toothed_belt_long_A_joint_{idx_magbot}" joint2="solid_connection_0_joint_{idx_magbot}"/>',
                f'\t\t<connect site1="pb_hinge_p_{idx_magbot}" site2="sc1_hinge_p_{idx_magbot}" solref="0.005 1" solimp="0.9 0.95 0.001"/>',
                f'\t\t<connect site1="pb_hinge_m_{idx_magbot}" site2="sc1_hinge_m_{idx_magbot}" solref="0.005 1" solimp="0.9 0.95 0.001"/>',
                f'\t\t<connect site1="m_{self.indices_mover_a[idx_magbot]}_rail_site" site2="rail_mount_1_site_{idx_magbot}" '
                'solref="0.005 1" solimp="0.9 0.95 0.001"/>',
                f'\t\t<connect site1="m_{self.indices_mover_b[idx_magbot]}_rail_site" site2="rail_mount_0_site_{idx_magbot}" '
                'solref="0.005 1" solimp="0.9 0.95 0.001"/>',
                '\t</equality>',
            ]
            equality_xml_str = '\n'.join(equality_list)
            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            if custom_outworldbody_xml_str is not None:
                custom_outworldbody_xml_str += equality_xml_str
            else:
                custom_outworldbody_xml_str = equality_xml_str
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str

        return custom_model_xml_strings

    def generate_platform_abRot_actuator_xml_strings(self) -> str:
        """Generate MuJoCo XML strings for the a,b-rot platform actuators of all 6D-Platform MagBots.

        :return: the MuJoCo XML actuator string (start <actuator> and end '</actuator>' are not included)
        """
        self.platform_ab_actuator_names = []
        actuator_str_list = []

        for idx_magbot in range(0, self.num_magbots):
            current_actuator_names = [f'platform_actuator_rot_a_{idx_magbot}', f'platform_actuator_rot_b_{idx_magbot}']
            self.platform_ab_actuator_names.append(current_actuator_names)
            actuator_str_list.append(f'\n\t\t<!-- platform A,B rot actuators MagBot {idx_magbot} -->')
            actuator_str_list.append(
                f'\t\t<general name="{current_actuator_names[0]}" joint="ball_socket_joint_{idx_magbot}" gear="1 0 0" ctrlrange="-1 1" '
                'ctrllimited="true"/>'
            )
            actuator_str_list.append(
                f'\t\t<general name="{current_actuator_names[1]}" joint="ball_socket_joint_{idx_magbot}" gear="0 1 0" ctrlrange="-1 1" '
                'ctrllimited="true"/>'
            )

        actuator_xml_str = '\n'.join(actuator_str_list)
        return actuator_xml_str

    def platformZPos2MoverDist(self, platform_z_d: np.ndarray, mover_z_d: np.ndarray) -> np.ndarray:
        """Calculate the desired mover distance from a desired platform z-position. This method is vectorized and can therefore calculate
        desired mover distances for multiple MagBots. Additionally this method takes the minimum and maximum possible mover distances into
        account, i.e. distances that are less (greater) than the minimum (maximum) distance are set to the minimum (maximum) distance.

        :param platform_z_d: the desired platform z-positions. Shape: (num_samples,) or (num_samples,1)
        :param mover_z_d: the desired mover z-pos (corresponds to the distance between the tile surface and the bottom of a mover).
            Shape: (num_samples,) or (num_samples,1)
        :return: the desired mover distances. Shape: (num_samples,) or (num_samples,1) depending on the input shapes
        """
        assert platform_z_d.shape == mover_z_d.shape
        mover_dist_d = 2 * (0.091 + np.sqrt(0.156**2 - np.power((platform_z_d - mover_z_d - 0.1278), 2)))
        mask_less_than_min = mover_dist_d < self.min_dist_movers_m
        mask_greater_than_max = mover_dist_d > self.max_dist_movers_m

        if np.sum(mask_less_than_min) > 0:
            mover_dist_d[mask_less_than_min] = self.min_dist_movers_m
            logger.warn(
                '[6D Platform MagBot]: At least one desired platform z-position requires a mover distance that is less than the minimum '
                + 'mover distance. These values are set to the minimum possible distance.'
            )

        if np.sum(mask_greater_than_max) > 0:
            mover_dist_d[mask_greater_than_max] = self.max_dist_movers_m
            logger.warn(
                '[6D Platform MagBot]: At least one desired platform z-position requires a mover distance that is greater than the '
                + 'maximum mover distance. These values are set to the maximum possible distance.'
            )

        return mover_dist_d

    def platformSetPose2MoverSetPose(
        self, platform_pose_d: np.ndarray, mover_z_d: np.ndarray | float, use_euler: bool = False
    ) -> np.ndarray:
        """Calculate the desired mover pose from a desired platform pose. This method is vectorized and can therefore calculate desired
        mover poses for multiple MagBots.

        :param platform_pose_d: the desired platform poses (rotation specified as Euler angles (xyz) in rad). Shape: (num_samples,6)
        :param mover_z_d: the desired mover z-pos (corresponds to the distance between the tile surface and the bottom of a mover).
            Can be a numpy array of shape (num_samples,) or a single float value (similar position for all movers).
        :param use_euler: whether to use quaternions (w,x,y,z) or euler angles (xyz) to represent the desired mover rotations, defaults
            to False
        :return: the desired mover poses. Shape: if ``use_euler==True``: (num_samples,2,6), else: (num_samples,2,7).
            [:,0,:]: poses for the mover that controls the platform's alpha-rotation; [:,1,:]: poses for the mover that controls the
            platform's beta-rotation
        """
        num_samples = platform_pose_d.shape[0]
        if isinstance(mover_z_d, float):
            mover_z_d = np.array([mover_z_d] * num_samples)

        mover_pose_euler_d = np.zeros((num_samples, 2, 6))
        mover_pose_euler_d[:, 0, 2] = mover_z_d
        mover_pose_euler_d[:, 1, 2] = mover_z_d
        mover_dist_d = self.platformZPos2MoverDist(platform_z_d=platform_pose_d[:, 2], mover_z_d=mover_z_d)

        # ensure max platform rot
        platform_pose_d[:, 3] = np.maximum(np.minimum(platform_pose_d[:, 3], self.max_platform_rot_ab), -self.max_platform_rot_ab)
        platform_pose_d[:, 4] = np.maximum(np.minimum(platform_pose_d[:, 4], self.max_platform_rot_ab), -self.max_platform_rot_ab)

        # calculate desired mover C rotation
        mover_pose_euler_d[:, 0, 5] = (-1) * self.gear_ratio_platform2mover_a * platform_pose_d[:, 3] + platform_pose_d[:, 5]
        mover_pose_euler_d[:, 1, 5] = (-1) * self.gear_ratio_platform2mover_b * platform_pose_d[:, 4] + platform_pose_d[:, 5]

        # calculate x-positions of the movers
        mover_pose_euler_d[:, 0, 0] = platform_pose_d[:, 0] - (mover_dist_d / 2) * np.cos(platform_pose_d[:, 5])
        mover_pose_euler_d[:, 1, 0] = platform_pose_d[:, 0] + (mover_dist_d / 2) * np.cos(platform_pose_d[:, 5])

        # calculate y-positions of the movers
        mover_pose_euler_d[:, 0, 1] = platform_pose_d[:, 1] - (mover_dist_d / 2) * np.sin(platform_pose_d[:, 5])
        mover_pose_euler_d[:, 1, 1] = platform_pose_d[:, 1] + (mover_dist_d / 2) * np.sin(platform_pose_d[:, 5])

        if not use_euler:
            # euler2quat
            mover_pose_quat_d = np.zeros((num_samples, 2, 7))
            mover_pose_quat_d[:, :, :3] = mover_pose_euler_d[:, :, :3]
            mover_pose_quat_d[:, :, 3:] = rotations_utils.vec_euler2quat(mover_pose_euler_d[:, :, 3:].reshape((-1, 3))).reshape(
                (num_samples, 2, 4)
            )
            return mover_pose_quat_d
        else:
            return mover_pose_euler_d

    def get_platform_pose(self, model: MjModel, data: MjData, indices_magbots: np.ndarray) -> np.ndarray:
        """Get the current pose of a MagBot's platform (measured at the center of the platform's surface). This method is vectorized
        and can therefore calculate the current pose of multiple MagBots.

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        :param indices_magbots: the indices of the MagBots for which to return the pose. Shape: (num_indices,)
        :return: the current pose of the platforms (rotation represented as quaternion). Shape: (num_indices,7)
            (representation: (x_p,y_p,z_p,w_o,x_o,y_o,z_o))
        """
        if not hasattr(self, 'platform_pose_site_ids'):
            self._update_cached_platform_site_ids(model)

        site_pose = np.zeros((indices_magbots.shape[0], 7))
        site_pose[:, :3] = data.site_xpos.reshape((-1, 3))[self.platform_pose_site_ids[indices_magbots], :]

        site_xmats = data.site_xmat.reshape((-1, 3, 3))[self.platform_pose_site_ids[indices_magbots], :, :]
        site_pose[:, 3:] = rotations_utils.vec_mat2quat(site_xmats)

        return site_pose

    def get_platform_vel(self, model: MjModel, data: MjData, idx_magbot: int) -> np.ndarray:
        """Get the current velocity of a MagBot's platform (measured at the center of the platform's surface).

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        :param idx_magbot: the index of the MagBot for which to return the velocity
        :return: the current velocity of the platform. Shape: (6,) (first 3: linear velocities, last 3: angular velocities)
        """
        if hasattr(self, 'platform_pose_site_ids'):
            site_id = self.platform_pose_site_ids[idx_magbot]
        else:
            site_id = model.site(self.platform_pose_site_names[idx_magbot]).id
        site_vel = np.zeros((6,))
        # first 3: angluar velcity, last 3: linear velocity
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, site_vel, 0)
        # switch linear and angluar velocities
        site_vel = site_vel[[3, 4, 5, 0, 1, 2]]
        return site_vel

    def get_platform_acc(self, model: MjModel, data: MjData, idx_magbot: int) -> np.ndarray:
        """Get the current acceleration of a MagBot's platform (measured at the center of the platform's surface).

        :param model: mjModel of the MuJoCo environment
        :param data: mjData of the MuJoCo environment
        :param idx_magbot: the index of the MagBot for which to return the acceleration
        :return: the current acceleration of the platform. Shape: (6,) (first 3: linear accelerations, last 3: angular accelerations)
        """
        if hasattr(self, 'platform_pose_site_ids'):
            site_id = self.platform_pose_site_ids[idx_magbot]
        else:
            site_id = model.site(self.platform_pose_site_names[idx_magbot]).id
        site_acc = np.zeros((6,))
        # first 3: angluar acceleration, last 3: linear acceleration
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, site_acc, 0)
        # switch linear and angluar accelerations
        site_acc = site_acc[[3, 4, 5, 0, 1, 2]]
        return site_acc
