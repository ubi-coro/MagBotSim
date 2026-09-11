import mujoco
import numpy as np

from magbotsim import BasicMagBotEnv, GripperMagBotsAPM4330, GripperMagBotsAPM4330Params, MoverImpedanceController


class GripperMagBotApplicationExampleEnv(BasicMagBotEnv):
    """Pick-and-place example environment with one Gripper MagBot."""

    def __init__(self) -> None:
        # product
        self.product_joint_name = 'product_joint'
        # mover parameters, collision parameters and initial start positions
        mover_mass = np.array([1.264] * 3)
        bumper_mass = np.array([0.05] * 3)
        mover_params = {
            'shape': 'mesh',
            'mesh': {'mover_stl_path': ['beckhoff_apm4330_mover'] * 3, 'bumper_stl_path': ['beckhoff_apm4330_bumper'] * 3},
            'mass': mover_mass - bumper_mass,
            'bumper_mass': bumper_mass,
        }
        collision_params = {'shape': 'box', 'size': np.array([[0.155 / 2, 0.155 / 2]] * 3) + 0.001}

        self.initial_m0m1_xy_dist = np.array([0.48, 0.0])  # [m]
        self.initial_m0m2_xy_dist = np.array([0.24, -0.12])  # [m]
        self.initial_mover_start_xy_pos = np.array(
            [
                # magbot 0
                [0.36, 0.6],
                [0.36, 0.6] + self.initial_m0m1_xy_dist,
                [0.36, 0.6] + self.initial_m0m2_xy_dist,
            ]
        )
        self.initial_mover_z_pos = 0.001

        # MagBots
        self.num_magbots = 1  # <-- number of MagBots in this environment
        self.indices_mover_c0 = np.array([0])  # <-- movers attached to connection 0
        self.indices_mover_c1 = np.array([1])  # <-- movers attached to connection 1
        self.indices_mover_c2 = np.array([2])  # <-- movers attached to connection 2
        self.magbot_params = GripperMagBotsAPM4330Params()

        # init BasicMagBotEnv
        super().__init__(
            layout_tiles=np.ones((5, 5)),
            num_movers=3,
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
                joint_mask=np.array([1, 1, 1, 1, 1, 0]),
                translational_stiffness=np.array([150.0, 150.0, 10.0]),
                rotational_stiffness=np.array([5.0, 5.0, 1.0]),
            )
            for mover_idx in range(self.num_movers)
        ]

        # reload model to add MagBots and actuators
        self.reload_model(mover_start_xy_pos=self.initial_mover_start_xy_pos)

    def _custom_xml_string_callback(self, custom_model_xml_strings: dict[str, str] | None = None) -> dict[str, str]:
        """Add the MagBots and actuators for the movers to the MuJoCo model by modifying the ``custom_model_xml_strings``-dict.

        :param custom_model_xml_strings: the current ``custom_model_xml_strings``-dict which is modified by this callback, defaults to None
        :return: the modified ``custom_model_xml_strings``-dict
        """
        if custom_model_xml_strings is None:
            custom_model_xml_strings = {}
            custom_model_xml_strings.update({'custom_mover_body_xml_str_list': [None] * self.num_movers})

        custom_model_xml_strings['custom_visual_xml_str'] = (
            '<visual><headlight ambient="0.2 0.2 0.2" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/></visual>'
        )

        if hasattr(self, 'mover_joint_names'):
            mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            # generate custom model XML strings for the two MagBots
            self.magbots = GripperMagBotsAPM4330(
                num_magbots=self.num_magbots,
                indices_mover_c0=self.indices_mover_c0,
                indices_mover_c1=self.indices_mover_c1,
                indices_mover_c2=self.indices_mover_c2,
                magbot_params=self.magbot_params,
                cycle_time=self.cycle_time,
            )
            custom_model_xml_strings = self.magbots.generate_magbot_xml_strings(
                initial_pos_xyz_mover_c0=mover_qpos[self.indices_mover_c0, :3], custom_model_xml_strings=custom_model_xml_strings
            )

            # add actuators
            actuator_lines = ['\n\n\t<actuator>']
            for idx_mover in range(0, self.num_movers):
                actuator_lines.append(self.impedance_controllers[idx_mover].generate_actuator_xml_string(idx_mover=idx_mover))
            # MagBot gripper and mover c actuators
            actuator_lines.append(self.magbots.generate_actuator_xml_strings())
            # join
            actuator_lines.append('\n\t</actuator>')
            custom_outworldbody_xml_str = custom_model_xml_strings.get('custom_outworldbody_xml_str', None)
            custom_model_xml_strings['custom_outworldbody_xml_str'] = custom_outworldbody_xml_str + ''.join(actuator_lines)

            # assets application
            custom_assets_xml_str = custom_model_xml_strings['custom_assets_xml_str']
            application_assets_list = [
                # extrusions
                '\n\t\t<!-- assets application example -->',
                '\t\t<mesh name="alu_extrusion_mesh" file="./misc/gripper_application_example/50x100_1250.stl" scale="1 1 1" />',
                '\t\t<mesh name="cover_mesh" file="./misc/gripper_application_example/cover.stl" scale="1 1 1" />',
                # platforms for product supply
                '\t\t<mesh name="supply_platform_high_mesh" file="./misc/gripper_application_example/supply_platform_high.stl" '
                'scale="1 1 1"/>',
                '\t\t<mesh name="supply_platform_low_mesh" file="./misc/gripper_application_example/supply_platform_low.stl" '
                'scale="1 1 1"/>',
            ]
            application_assets_xml_str = '\n'.join(application_assets_list)
            custom_assets_xml_str += application_assets_xml_str
            custom_model_xml_strings['custom_assets_xml_str'] = custom_assets_xml_str

            custom_worldbody_xml_str = custom_model_xml_strings['custom_worldbody_xml_str']
            application_bodies_list = [
                # extrusions
                '\n\t\t<!-- aluminum extrusions and covers -->',
                '\t\t<body name="aluminum_extrusions" pos="0 0 0" gravcomp="1">',
                '\t\t\t<geom name="alu_extrusion_0" type="mesh" mesh="alu_extrusion_mesh" mass="1.0" pos="0 -0.05 -0.04" '
                'euler="0 0 -1.5708" material="gray"/>',
                '\t\t\t<geom name="alu_extrusion_1" type="mesh" mesh="alu_extrusion_mesh" mass="1.0" pos="0 -0.05 -0.04" material="gray"/>',
                '\t\t\t<geom name="alu_extrusion_2" type="mesh" mesh="alu_extrusion_mesh" mass="1.0" pos="-0.05 1.2 -0.04" '
                'euler="0 0 -1.5708" material="gray"/>',
                '\t\t\t<geom name="alu_extrusion_3" type="mesh" mesh="alu_extrusion_mesh" mass="1.0" pos="1.25 0 -0.04" material="gray"/>',
                '\t\t\t<geom name="cover_geom_0" type="mesh" mesh="cover_mesh" mass="0.1" pos="-0.05 -0.051 -0.04" euler="0 0 0" '
                'material="black"/>',
                '\t\t\t<geom name="cover_geom_1" type="mesh" mesh="cover_mesh" mass="0.1" pos="1.2 1.251 -0.04" euler="0 0 0" '
                'material="black"/>',
                '\t\t\t<geom name="cover_geom_2" type="mesh" mesh="cover_mesh" mass="0.1" pos="1.251 -0.05 -0.04" euler="0 0 1.5708" '
                'material="black"/>',
                '\t\t\t<geom name="cover_geom_3" type="mesh" mesh="cover_mesh" mass="0.1" pos="-0.05 1.201 -0.04" euler="0 0 1.5708" '
                'material="black"/>',
                '\t\t</body>',
                # platform for product supply
                '\t\t<body name="supply_platform_high" pos="0.72 -0.025 0.06" euler="0 0 3.141" gravcomp="1">',
                '\t\t\t<geom name="supply_platform_high_geom" type="mesh" mesh="supply_platform_high_mesh" mass="0.3" material="black"/>',
                '\t\t</body>',
                '\t\t<body name="supply_platform_low" pos="0.48 1.225 0.052" euler="0 0 0" gravcomp="1">',
                '\t\t\t<geom name="supply_platform_low_geom" type="mesh" mesh="supply_platform_low_mesh" mass="0.3" material="black"/>',
                '\t\t</body>',
                # product
                '\t\t<body name="product" pos="0.48 1.17075 0.157" euler="0 0 0">',
                '\t\t\t<geom name="product_bottom_geom" type="box" size="0.008 0.008 0.004" pos="0 0 -0.004" mass="0.002" '
                'material="white" friction="1.0 0.05 0.005" condim="4" priority="4"/>',
                '\t\t\t<geom name="product_top_geom" type="box" size="0.008 0.008 0.004" pos="0 0 0.004" mass="0.002" '
                'material="red" friction="1.0 0.05 0.005" condim="4" priority="4"/>',
                f'\t\t\t<joint name="{self.product_joint_name}" type="free"/>',
                '\t\t</body>',
            ]
            application_bodies_xml_str = '\n'.join(application_bodies_list)
            custom_worldbody_xml_str += application_bodies_xml_str
            custom_model_xml_strings['custom_worldbody_xml_str'] = custom_worldbody_xml_str

            custom_model_xml_strings['custom_option_xml_str'] = (
                '\t<option timestep="0.001" cone="elliptic" jacobian="dense" gravity="0 0 -9.81" integrator="implicitfast" '
                'impratio="10" noslip_tolerance="1e-8"><flag diagexact="enable"/></option>'
            )

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray) -> None:
        """Generate a new model XML string with new start positions for movers.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover
        """
        assert np.allclose(mover_start_xy_pos[1, :] - mover_start_xy_pos[0, :], self.initial_m0m1_xy_dist)
        assert np.allclose(mover_start_xy_pos[2, :] - mover_start_xy_pos[0, :], self.initial_m0m2_xy_dist)
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
        self.magbots.update_cached_mujoco_data(self.model)

        # render the environment after reloading
        if self.render_mode is not None:
            self.viewer_collection.reload_model(self.model, self.data)
        self.render()

    def step(
        self, gripper_setposes: np.ndarray, single_track: np.ndarray | bool, num_cycles: int = 40, render_every_cycle: bool = False
    ) -> None:
        """Perform one or multiple simulation steps, including caluclation of mover controls, MuJoCo integrator steps, rendering, and
        collision checking.

        :param gripper_setposes: the desired target positions of the MagBot grippers (numpy array of shape (num_magbots, 7) using Euler
            angles (xyz) in rad). The 7th entry is the desired distance between the two gripper fingers in m.
        :param single_track: whether to calculate the mover set positions in single-track mode, defaults to False
        :param num_cycles: the number of control cycles for which to apply the same controls, defaults to 40
        :param render_every_cycle: whether to call ``render()`` after each integrator step in the ``step()`` method, defaults to
            False. Rendering every cycle leads to a smoother visualization of the scene, but can also be computationally expensive.
            Thus, this parameter provides the possibility to speed up training and evaluation. Regardless of this parameter, the scene
            is always rendered after ``num_cycles``.
        """
        assert gripper_setposes.shape == (self.num_magbots, 7)

        for _ in range(0, num_cycles):
            # calculate controls
            # gripper set pose to mover set pose
            mover_poses = self.magbots.gripperSetPose2MoverSetPose(
                gripper_pose_d=gripper_setposes, z_m=self.initial_mover_zpos, single_track=single_track
            )
            mover_poses = mover_poses.reshape((-1, 6))
            # update controls for mover impedance controllers
            for idx_mover in range(0, self.num_movers):
                self.impedance_controllers[idx_mover].update(
                    model=self.model,
                    data=self.data,
                    pos_d=mover_poses[idx_mover, :3],
                    quat_d=np.array([1, 0, 0, 0]),
                    additional_mass=self.magbots.magbot_masses[0] / 2,
                )
            # control gripper
            current_mover_qpos = self.get_mover_qpos(mover_names=self.mover_names, add_noise=False)
            current_mover_qvels = self.get_mover_qvel(mover_names=self.mover_names, add_noise=False)

            self.magbots.control_gripper(
                model=self.model,
                data=self.data,
                mover_poses_d=mover_poses,
                current_mover_qpos=current_mover_qpos,
                current_mover_qvels=current_mover_qvels,
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
