import numpy as np
import mujoco
from magbotsim import BasicMagBotEnv, SixDPlatformMagBotsAPM4330, MoverImpedanceController


class SixDPlatformMagBotApplicationExampleEnv(BasicMagBotEnv):
    """Application example environment with one 6D-Platform MagBot. The task is to pick up a product from a supply station and
    place it in a container.
    """

    def __init__(self) -> None:
        # product
        self.product_joint_name = 'product_joint'
        # mover parameters, collision parameters and initial start positions
        mover_mass = np.array([1.264] * 2)
        bumper_mass = np.array([0.05] * 2)
        mover_params = {
            'shape': 'mesh',
            'mesh': {'mover_stl_path': ['beckhoff_apm4330_mover'] * 2, 'bumper_stl_path': ['beckhoff_apm4330_bumper'] * 2},
            'mass': mover_mass - bumper_mass,
            'bumper_mass': bumper_mass,
        }
        # we model the mover as a simple 2D box to check collisions
        # since we are using Beckhoff XPlanar movers (APM4330), we
        # use the sizes specified in the technical drawings plus a safety margin
        collision_params = {'shape': 'box', 'size': np.array([[0.155 / 2, 0.155 / 2]] * 2) + 0.001}

        self.initial_mover_x_dist = 0.4027  # [m]
        self.initial_mover_start_xy_pos = np.array(
            [
                [0.36, 0.36],
                [0.36 + self.initial_mover_x_dist, 0.36],
            ]
        )
        self.initial_mover_z_pos = 0.001
        self.indices_mover_a = np.array([0])
        self.indices_mover_b = np.array([1])

        # MagBots
        self.num_magbots = 1

        # init BasicMagBotEnv
        super().__init__(
            layout_tiles=np.ones((4, 3)),
            num_movers=2,
            mover_params=mover_params,
            initial_mover_zpos=self.initial_mover_z_pos,
            table_height=0.2,
            collision_params=collision_params,
            initial_mover_start_xy_pos=self.initial_mover_start_xy_pos,
            custom_model_xml_strings=None,
            use_mj_passive_viewer=True,
        )

        # impedance controller
        self.impedance_controllers = [
            MoverImpedanceController(
                model=self.model,
                mover_joint_name=self.mover_joint_names[mover_idx],
                mover_half_height=self.mover_size[mover_idx, 2],
                joint_mask=np.array([1, 1, 1, 1, 1, 1]),
                translational_stiffness=np.array([50.0, 50.0, 50.0]),
                rotational_stiffness=np.array([1.0, 1.0, 5.0]),
            )
            for mover_idx in range(self.num_movers)
        ]

        # reload model to add MagBots and actuators
        self.reload_model(mover_start_xy_pos=self.initial_mover_start_xy_pos)

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str]:
        """Add the MagBot, product, supply station, and container, as well as actuators by modifying the ``custom_model_xml_strings``-dict.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback, defaults to None
        :return: the modified ``custom_model_xml_strings``-dict
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
            custom_model_xml_strings.update({'custom_mover_body_xml_str_list': [None] * self.num_movers})

        if hasattr(self, 'mover_joint_names'):
            mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            # generate custom model XML strings for the MagBot
            self.magbot = SixDPlatformMagBotsAPM4330(
                num_magbots=self.num_magbots, indices_mover_a=self.indices_mover_a, indices_mover_b=self.indices_mover_b
            )
            custom_model_xml_strings = self.magbot.generate_magbot_xml_strings(
                initial_pos_xyz_mover_b=mover_qpos[self.indices_mover_b, :3], custom_model_xml_strings=custom_model_xml_strings
            )

            # add actuators
            actuator_lines = ['\n\n\t<actuator>']
            # mover actuators
            for idx_mover in range(0, self.num_movers):
                actuator_lines.append(f'\n\t\t<!-- actuators mover {idx_mover} -->')
                actuator_lines.append(self.impedance_controllers[idx_mover].generate_actuator_xml_string(idx_mover=idx_mover))
                actuator_lines.append('\n')
            # MagBot platform a,b rot actuators
            actuator_lines.append(self.magbot.generate_platform_abRot_actuator_xml_strings())
            # join
            actuator_lines.append('\n\t</actuator>')
            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str + ''.join(actuator_lines)

            # assets application
            custom_assets_xml_str = custom_model_xml_strings['custom_assets_xml_str']
            application_assets_list = [
                # extrusions
                '\n\t\t<!-- assets application example -->',
                '\t\t<mesh name="alu_extrusion_short_mesh" file="./misc/6D_platform_application_example/50x100_770.STL" scale="1 1 1" />',
                '\t\t<mesh name="alu_extrusion_long_mesh" file="./misc/6D_platform_application_example/50x100_1010.STL" scale="1 1 1" />',
                '\t\t<mesh name="cover_mesh" file="./misc/6D_platform_application_example/cover.STL" scale="1 1 1" />',
                # containter, supply station, and product
                '\t\t<mesh name="product_mesh" file="./misc/6D_platform_application_example/product.STL" scale="1 1 1" />',
                '\t\t<mesh name="supply_rack_left_mesh" file="./misc/6D_platform_application_example/supply_rack_left.STL" scale="1 1 1"/>',
                '\t\t<mesh name="supply_rack_right_mesh" file="./misc/6D_platform_application_example/supply_rack_right.STL" '
                'scale="1 1 1"/>',
                '\t\t<mesh name="spacer_mesh" file="./misc/6D_platform_application_example/spacer.STL" scale="1 1 1" />',
                '\t\t<mesh name="container_front_mesh" file="./misc/6D_platform_application_example/container_front.STL" scale="1 1 1" />',
                '\t\t<mesh name="container_side_mesh" file="./misc/6D_platform_application_example/container_side.STL" scale="1 1 1" />',
                '\t\t<mesh name="container_back_mesh" file="./misc/6D_platform_application_example/container_back.STL" scale="1 1 1" />',
            ]
            application_assets_xml_str = '\n'.join(application_assets_list)
            custom_assets_xml_str += application_assets_xml_str
            custom_model_xml_strings['custom_assets_xml_str'] = custom_assets_xml_str

            custom_worldbody_xml_str = custom_model_xml_strings['custom_worldbody_xml_str']
            application_bodies_list = [
                # extrusions
                '\n\t\t<!-- aluminum extrusions and covers -->',
                '\t\t<body name="aluminum_extrusions" pos="0 0 0" gravcomp="1">',
                '\t\t\t<geom name="alu_extrusion_0" type="mesh" mesh="alu_extrusion_long_mesh" mass="2.5" pos="0 -0.05 -0.04" '
                'material="gray"/>',
                '\t\t\t<geom name="alu_extrusion_1" type="mesh" mesh="alu_extrusion_short_mesh" mass="1.5" pos="-0.05 -0.05 -0.04" '
                'material="gray"/>',
                '\t\t\t<geom name="alu_extrusion_2" type="mesh" mesh="alu_extrusion_long_mesh" mass="2.5" pos="-0.05 0.72 -0.04" '
                'material="gray"/>',
                '\t\t\t<geom name="alu_extrusion_3" type="mesh" mesh="alu_extrusion_short_mesh" mass="1.5" pos="0.96 0 -0.04" '
                'material="gray"/>',
                '\t\t\t<geom name="cover_geom_0" type="mesh" mesh="cover_mesh" mass="0.5" pos="-0.05 -0.051 -0.04" euler="0 0 0" '
                'material="black"/>',
                '\t\t\t<geom name="cover_geom_1" type="mesh" mesh="cover_mesh" mass="0.5" pos="0.96 0.771 -0.04" euler="0 0 0" '
                'material="black"/>',
                '\t\t\t<geom name="cover_geom_2" type="mesh" mesh="cover_mesh" mass="0.5" pos="1.011 -0.05 -0.04" euler="0 0 1.5708" '
                'material="black"/>',
                '\t\t\t<geom name="cover_geom_3" type="mesh" mesh="cover_mesh" mass="0.5" pos="-0.05 0.72 -0.04" euler="0 0 1.5708" '
                'material="black"/>',
                '\t\t</body>',
                # containter, supply station, and product
                '\n\t\t<!-- supply rack -->',
                '\t\t<body name="supply_rack" pos="0.66518 0.0055 0.03" gravcomp="1" euler="0 0 3.1416">',
                '\t\t\t<geom name="spacer_geom" type="mesh" mesh="spacer_mesh" mass="0.5" pos="0 0 0" material="black" contype="0" '
                'conaffinity="0"/>',
                '\t\t\t<geom name="supply_rack_left_geom" type="mesh" mesh="supply_rack_left_mesh" mass="1.0" pos="0 0 0.0835" '
                'material="black" contype="0" conaffinity="0"/>',
                '\t\t\t<geom name="supply_rack_right_geom" type="mesh" mesh="supply_rack_right_mesh" mass="1.0" pos="0.11 0.0 0.0835" '
                'euler= "0 0 1.5708" material="black" contype="0" conaffinity="0"/>',
                '\t\t</body>',
                '\t\t<body name="holding_rack"pos="0.66518 0.0055 0.03" gravcomp="1" euler="0 0 3.1416">',
                '\t\t\t<geom name="left_pole" type="box" size="0.0025 0.03775 0.002" mass="0.2" pos="0.0025 -0.06175 0.2054" '
                'material="black"/>',
                '\t\t\t<geom name="right_pole" type="box" size="0.0025 0.03775 0.002" mass="0.2" pos="0.1075 -0.06175 0.2054" '
                'material="black"/>',
                '\t\t\t<geom name="left_front_pole" type="box" size="0.0025 0.002 0.007" mass="0.2" pos="0.0025 -0.1014 0.2105" '
                'material="black"/>',
                '\t\t\t<geom name="right_front_pole" type="box" size="0.0025 0.002 0.007" mass="0.2" pos="0.1075 -0.1014 0.2105" '
                'material="black"/>',
                '\t\t</body>',
                '\n\t\t<!-- container -->',
                '\t\t<body name="product_container" pos="0.440 0.7755 -0.037" gravcomp="1" euler="0 0 3.1416">',
                '\t\t\t<geom name="container_front" type="mesh" mesh="container_front_mesh" mass="0.5" pos="0.0 0.0 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_back" type="mesh" mesh="container_back_mesh" mass="0.5" pos="0.0 -0.153 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_side_left" type="mesh" mesh="container_side_mesh" mass="0.5" pos="0.0 -0.15 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_side_right" type="mesh" mesh="container_side_mesh" mass="0.5" pos="0.197 -0.15 0.0" '
                'material="black"/>',
                '\t\t\t<geom name="container_bottom" type="box" size="0.1 0.079 0.0025" mass="0.5" pos="0.1 -0.074 -0.002" '
                'material="black"/>',
                '\t\t</body>',
                '\n\t\t<!-- product -->',
                '\t\t<body name="product_body" pos="0.61019 0.0673 0.22752" gravcomp="0">',
                '\t\t\t<geom name="product_geom_0" type="box" size="0.0495 0.03205 0.01" mass="0.06" pos="0 0 0" euler="0 0 0" '
                'material="white" solref="0.005 1" priority="1" friction="0.24 0.0001 0.00001" condim="6"/>',
                '\t\t\t<geom name="product_geom_1" type="box" size="0.055 0.0375 0.0015" mass="0.004" pos="0 0 0.0115" euler="0 0 0" '
                'material="white"/>',
                f'\t\t\t<joint name="{self.product_joint_name}" type="free"/>',
                '\t\t</body>',
            ]
            application_bodies_xml_str = '\n'.join(application_bodies_list)
            custom_worldbody_xml_str += application_bodies_xml_str
            custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

            custom_model_xml_strings['custom_option_xml_str'] = (
                '\n\t<option timestep="0.001" cone="elliptic" jacobian="auto" gravity="0 0 -9.81" impratio="10" noslip_tolerance="1e-8" '
                'solver="Newton"/>'
            )

        return custom_model_xml_strings

    def reset_product_position(self) -> None:
        """Reset the position of the product to its initial position in the supply station."""
        self.data.qpos[self.product_joint_qpos_adr : self.product_joint_qpos_adr + 7] = self.initial_product_joint_qpos
        mujoco.mj_forward(self.model, self.data)

    def reload_model(self, mover_start_xy_pos: np.ndarray) -> None:
        """Generate a new model XML string with new start positions for movers.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover
        """
        assert np.allclose(mover_start_xy_pos[1, 0] - mover_start_xy_pos[0, 0], self.initial_mover_x_dist)
        # generate a new model XML string
        custom_model_xml_strings = self._custom_xml_string_callback(custom_model_xml_strings=self.custom_model_xml_strings_before_cb)
        model_xml_str = self.generate_model_xml_string(
            mover_start_xy_pos=mover_start_xy_pos, mover_goal_xy_pos=None, custom_xml_strings=custom_model_xml_strings
        )
        # compile the MuJoCo model
        self.model = mujoco.MjModel.from_xml_string(model_xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # update cached mujoco data
        self.update_cached_mover_mujoco_data()
        for idx_mover in range(0, self.num_movers):
            self.impedance_controllers[idx_mover].update_cached_mujoco_data(self.model)
        self.magbot.update_cached_mujoco_data(self.model)

        # product joint
        product_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.product_joint_name)
        self.product_joint_qpos_adr = self.model.jnt_qposadr[product_joint_id]
        self.initial_product_joint_qpos = self.data.qpos[self.product_joint_qpos_adr : self.product_joint_qpos_adr + 7].copy()

        # render the environment after reloading
        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)
        self.render()

    def step(self, platform_setposes: np.ndarray, num_cycles: int = 40, render_every_cycle: bool = False) -> None:
        """Perform one or multiple simulation steps, including caluclation of mover controls, MuJoCo integrator steps, rendering, and
        collision checking.

        :param platform_setposes: the desired target positions of the MagBot platforms (numpy array of shape (num_magbots, 6) using Euler
            angles (xyz) in rad)
        :param num_cycles: the number of control cycles for which to apply the same controls, defaults to 40
        :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
            False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
            Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
            is always rendered after ``num_cycles``.
        """
        assert platform_setposes.shape == (self.num_magbots, 6)

        for _ in range(0, num_cycles):
            # calculate controls
            # platform set pose to mover set pose
            mover_poses = self.magbot.platformSetPose2MoverSetPose(
                platform_pose_d=platform_setposes, mover_z_d=self.initial_mover_zpos, use_euler=False
            )
            # update controls for mover impedance controllers
            mover_poses_tmp = mover_poses.reshape((self.num_movers, 7))
            for idx_mover in range(0, self.num_movers):
                self.impedance_controllers[idx_mover].update(
                    model=self.model,
                    data=self.data,
                    pos_d=mover_poses_tmp[idx_mover, :3],
                    quat_d=mover_poses_tmp[idx_mover, 3:],
                    additional_mass=self.magbot.magbot_masses[0] / 2,
                )
            # coupling: mover rotation <-> platform rotation
            current_mover_poses = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            self.magbot.control_platform_ab_rot(
                model=self.model,
                data=self.data,
                mover_a_quats=current_mover_poses[self.indices_mover_a, 3:],
                mover_b_quats=current_mover_poses[self.indices_mover_b, 3:],
            )
            # integration
            mujoco.mj_step(self.model, self.data, nstep=1)
            # render every cycle for a smooth visualization of the movement
            if render_every_cycle:
                self.render()
            # check wall and mover collision every cycle to ensure that the collisions are detected and
            # all intermediate mover positions are valid and without collisions
            wall_collision = self.check_wall_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            ).any()
            mover_collision = self.check_mover_collision(
                mover_names=self.mover_names,
                c_size=self.c_size,
                add_safety_offset=False,
                mover_qpos=None,
                add_qpos_noise=True,  # would also occur in a real system
            )
            if mover_collision or wall_collision:
                break
        self.render()


if __name__ == '__main__':
    platform_setpose = np.array([[0.56133, 0.36, 0.205, 0.0, 0.0, 0.0]])

    env = SixDPlatformMagBotApplicationExampleEnv()

    num_cycles = 5
    cnt = 0
    err_scale_xy_fast = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    err_scale_xyc_fast = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 6.0])
    env.impedance_controllers[0].set_pose_err_scale(err_scale_xy_fast)
    env.impedance_controllers[1].set_pose_err_scale(err_scale_xy_fast)
    try:
        while True:
            platform_pose = env.magbot.get_platform_pose(env.model, env.data, np.array([0]))

            pos_err = np.abs(platform_setpose[:, :3] - platform_pose[:, :3])
            if (pos_err < 1e-2).all():
                if cnt == 0:
                    env.impedance_controllers[0].set_pose_err_scale(err_scale_xy_fast)
                    env.impedance_controllers[1].set_pose_err_scale(err_scale_xy_fast)
                    platform_setpose = np.array([[0.61019, 0.133, 0.205, 0.0, 0.0, 0.0]])
                elif cnt == 1:
                    platform_setpose[0, 2] = 0.24
                elif cnt == 2:
                    env.impedance_controllers[0].set_pose_err_scale(1.0)
                    env.impedance_controllers[1].set_pose_err_scale(1.0)
                    platform_setpose[0, 1] = 0.222
                elif cnt == 3:
                    env.impedance_controllers[0].set_pose_err_scale(0.5)
                    env.impedance_controllers[1].set_pose_err_scale(0.5)
                    platform_setpose[0, :3] = np.array([0.3519234, 0.6, 0.205])
                elif cnt == 4:
                    env.impedance_controllers[0].set_pose_err_scale(err_scale_xyc_fast)
                    env.impedance_controllers[1].set_pose_err_scale(err_scale_xyc_fast)
                    platform_setpose[0, 3] = np.deg2rad(-14.0)
                elif cnt == 500:
                    platform_setpose[0, 3] = 0.0

                if cnt < 700:
                    cnt += 1
                else:
                    cnt = 0
                    env.impedance_controllers[0].set_pose_err_scale(err_scale_xy_fast)
                    env.impedance_controllers[1].set_pose_err_scale(err_scale_xy_fast)
                    platform_setpose = np.array([[0.56133, 0.36, 0.205, 0.0, 0.0, 0.0]])
                    env.reset_product_position()

            env.step(platform_setposes=platform_setpose, num_cycles=num_cycles, render_every_cycle=False)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
