import numpy as np
import mujoco
from magbotsim import BasicMagBotEnv, MoverImpedanceController, GripperMagBotsAPM4330, GripperMagBotsAPM4330Params


class GripperMagBotExampleEnv(BasicMagBotEnv):
    """Simple example environment with two Gripper MagBots."""

    def __init__(self) -> None:
        # mover parameters, collision parameters and initial start positions
        mover_mass = np.array([1.264] * 6)
        bumper_mass = np.array([0.05] * 6)
        mover_params = {
            'shape': 'mesh',
            'mesh': {'mover_stl_path': ['beckhoff_apm4330_mover'] * 6, 'bumper_stl_path': ['beckhoff_apm4330_bumper'] * 6},
            'mass': mover_mass - bumper_mass,
            'bumper_mass': bumper_mass,
        }
        collision_params = {'shape': 'box', 'size': np.array([[0.155 / 2, 0.155 / 2]] * 6) + 0.001}

        self.initial_m0m1_xy_dist = np.array([0.48, 0.0])  # [m]
        self.initial_m0m2_xy_dist = np.array([0.24, -0.12])  # [m]
        self.initial_mover_start_xy_pos = np.array(
            [
                # magbot 0
                [0.24, 0.36],
                [0.24, 0.36] + self.initial_m0m1_xy_dist,
                [0.24, 0.36] + self.initial_m0m2_xy_dist,
                # magbot 1
                [1.2, 0.36],
                [1.2, 0.36] + self.initial_m0m1_xy_dist,
                [1.2, 0.36] + self.initial_m0m2_xy_dist,
            ]
        )
        self.initial_mover_z_pos = 0.001

        # MagBots
        self.num_magbots = 2  # <-- number of MagBots in this environment
        self.indices_mover_c0 = np.array([0, 3])  # <-- movers attached to connection 0
        self.indices_mover_c1 = np.array([1, 4])  # <-- movers attached to connection 1
        self.indices_mover_c2 = np.array([2, 5])  # <-- movers attached to connection 2
        self.magbot_params = GripperMagBotsAPM4330Params()

        # init BasicMagBotEnv
        super().__init__(
            layout_tiles=np.ones((8, 3)),
            num_movers=6,
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
                rotational_stiffness=np.array([5.0, 5.0, 0.0]),
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

        return custom_model_xml_strings

    def reload_model(self, mover_start_xy_pos: np.ndarray) -> None:
        """Generate a new model XML string with new start positions for movers.

        :param mover_start_xy_pos: None or a numpy array of shape (num_movers,2) containing the (x,y) starting positions of each mover
        """
        assert np.allclose(mover_start_xy_pos[1, :] - mover_start_xy_pos[0, :], self.initial_m0m1_xy_dist)
        assert np.allclose(mover_start_xy_pos[4, :] - mover_start_xy_pos[3, :], self.initial_m0m1_xy_dist)
        assert np.allclose(mover_start_xy_pos[2, :] - mover_start_xy_pos[0, :], self.initial_m0m2_xy_dist)
        assert np.allclose(mover_start_xy_pos[5, :] - mover_start_xy_pos[3, :], self.initial_m0m2_xy_dist)
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


if __name__ == '__main__':
    magbot_params = GripperMagBotsAPM4330Params()

    z_offset = (magbot_params.z_p_max - 0.01 - magbot_params.z_p_min) / 2
    fR = 0.08  # radius of the circle
    fTime = 0.0
    single_track = np.array([False, True])

    initial_z_pos = z_offset + magbot_params.z_p_min + magbot_params.n + 0.001
    # the 7th entry is the desired distance between the two gripper fingers in m
    finger_distance = 0.0
    gripper_setposes = np.array(
        [
            [0.48 + fR, 0.36, initial_z_pos, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), finger_distance],  # default mode
            [1.2 + 0.24 + fR, 0.48, initial_z_pos, np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0), finger_distance],  # single-track mode
        ]
    )
    env = GripperMagBotExampleEnv()

    num_cycles = 20
    try:
        while True:
            # xy circle + z sin movement
            gripper_setposes[:, 0] = np.array([0.48, 1.44]) + fR * np.cos(fTime)
            gripper_setposes[:, 1] = np.array([0.36, 0.36]) + fR * np.sin(fTime)
            gripper_setposes[:, 2] = z_offset * np.sin(2.0 * fTime) + initial_z_pos

            env.step(gripper_setposes=gripper_setposes, single_track=single_track, num_cycles=num_cycles, render_every_cycle=False)
            fTime += 0.001 * num_cycles
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
